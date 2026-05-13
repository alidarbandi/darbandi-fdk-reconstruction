from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np
from scipy.ndimage import shift as ndi_shift


@dataclass
class GeometryParams:
    effective_pixel_size_um: float
    DSO_mm: float
    DSD_mm: float
    Nx: int
    Ny: int
    Nz: int
    voxel_size_um: float
    object_offset_x_mm: float = 0.0
    object_offset_y_mm: float = 0.0
    object_offset_z_mm: float = 0.0
    detector_offset_vertical_px: float = 0.0
    detector_offset_horizontal_px: float = 0.0
    enable_center_offset: bool = False
    center_offset_px: float = 0.0
    center_offset_method: str = "detector_offset"
    center_shift_sign: float = 1.0


@dataclass
class ReconstructionResult:
    volume: np.ndarray
    geometry_report: list[str]
    started_at: datetime
    finished_at: datetime


def tigre_available() -> tuple[bool, str]:
    try:
        import tigre  # noqa: F401
        from tigre.algorithms import fdk  # noqa: F401
    except Exception as exc:
        return False, str(exc)
    return True, "TIGRE import succeeded."


def validate_geometry_params(params: GeometryParams) -> None:
    if params.effective_pixel_size_um <= 0:
        raise ValueError("Effective pixel size must be positive.")
    if params.voxel_size_um <= 0:
        raise ValueError("Voxel size must be positive.")
    if params.DSO_mm <= 0 or params.DSD_mm <= 0:
        raise ValueError("DSO and DSD must be positive.")
    if params.DSD_mm <= params.DSO_mm:
        raise ValueError("DSD must be larger than DSO for cone-beam geometry.")
    if params.Nx <= 0 or params.Ny <= 0 or params.Nz <= 0:
        raise ValueError("Reconstruction voxel counts Nx, Ny, and Nz must be positive.")
    if params.center_offset_method not in {"detector_offset", "image_shift"}:
        raise ValueError("Center offset method must be detector_offset or image_shift.")
    if params.center_shift_sign not in (-1, 1, -1.0, 1.0):
        raise ValueError("Center-shift sign convention must be +1 or -1.")


def computed_detector_pixel_mm(params: GeometryParams) -> float:
    p_eff_mm = params.effective_pixel_size_um * 1e-3
    magnification = params.DSD_mm / params.DSO_mm
    # The Zeiss optical/camera magnification is already baked into p_eff.
    # TIGRE needs the effective detector-plane sampling: p_eff * DSD/DSO.
    return p_eff_mm * magnification


def build_tigre_geometry(params: GeometryParams, detector_rows: int, detector_cols: int):
    validate_geometry_params(params)
    try:
        import tigre
    except Exception as exc:  # pragma: no cover - depends on local TIGRE install
        raise RuntimeError(
            "TIGRE is not importable. Install a compatible TIGRE build for this Python/CUDA setup."
        ) from exc

    try:
        geo = tigre.geometry(mode="cone", default=False)
    except TypeError:
        geo = tigre.geometry(mode="cone")

    d_detector_mm = computed_detector_pixel_mm(params)
    d_voxel_mm = params.voxel_size_um * 1e-3

    geo.DSD = float(params.DSD_mm)
    geo.DSO = float(params.DSO_mm)
    geo.accuracy = 0.5

    # Detector stack axis order in this app is (projection, v row, u column).
    # TIGRE Python examples commonly define detector arrays as [v, u].
    geo.nDetector = np.array([detector_rows, detector_cols], dtype=np.int32)
    geo.dDetector = np.array([d_detector_mm, d_detector_mm], dtype=np.float64)
    geo.sDetector = geo.nDetector * geo.dDetector

    # The GUI labels reconstruction size as Nx, Ny, Nz. We set TIGRE's voxel
    # arrays in z, y, x order so the returned volume is addressed as [z, y, x].
    # This follows the project prompt and keeps the displayed Z-slice intuitive.
    geo.nVoxel = np.array([params.Nz, params.Ny, params.Nx], dtype=np.int32)
    geo.dVoxel = np.array([d_voxel_mm, d_voxel_mm, d_voxel_mm], dtype=np.float64)
    geo.sVoxel = geo.nVoxel * geo.dVoxel

    geo.offOrigin = np.array(
        [params.object_offset_z_mm, params.object_offset_y_mm, params.object_offset_x_mm],
        dtype=np.float64,
    )

    off_v_mm = params.detector_offset_vertical_px * d_detector_mm
    off_u_mm = params.detector_offset_horizontal_px * d_detector_mm
    center_offset_mm = 0.0
    if params.enable_center_offset and params.center_offset_method == "detector_offset":
        center_offset_mm = params.center_shift_sign * params.center_offset_px * d_detector_mm
        off_u_mm += center_offset_mm
    # offDetector is stored as [vertical detector offset, horizontal detector offset].
    # The center-offset sign is sample/setup dependent; the GUI accepts positive
    # and negative values so the user can perform a sign test.
    geo.offDetector = np.array([off_v_mm, off_u_mm], dtype=np.float64)
    _force_tigre_size_consistency(geo)

    report = geometry_report(params, detector_rows, detector_cols, d_detector_mm, off_v_mm, off_u_mm, center_offset_mm)
    return geo, report


def _force_tigre_size_consistency(geo) -> None:
    # Some TIGRE builds perform a strict consistency check. Keep these values
    # in float64 and compute total sizes from the exact expression TIGRE checks.
    geo.dDetector = np.asarray(geo.dDetector, dtype=np.float64)
    geo.nDetector = np.asarray(geo.nDetector, dtype=np.int32)
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.dVoxel = np.asarray(geo.dVoxel, dtype=np.float64)
    geo.nVoxel = np.asarray(geo.nVoxel, dtype=np.int32)
    geo.sVoxel = geo.nVoxel * geo.dVoxel


def geometry_report(
    params: GeometryParams,
    detector_rows: int,
    detector_cols: int,
    d_detector_mm: float,
    off_v_mm: float,
    off_u_mm: float,
    center_offset_mm: float,
    angles_rad: np.ndarray | None = None,
    projection_stack: np.ndarray | None = None,
    fdk_filter: str = "ram_lak",
    angle_sign: int = 1,
) -> list[str]:
    magnification = params.DSD_mm / params.DSO_mm
    p_eff_mm = params.effective_pixel_size_um * 1e-3
    detector_fov_v_mm = detector_rows * d_detector_mm
    detector_fov_u_mm = detector_cols * d_detector_mm
    object_fov_v_mm = detector_rows * p_eff_mm
    object_fov_u_mm = detector_cols * p_eff_mm

    if angles_rad is None or len(angles_rad) == 0:
        number_of_angles = "unavailable"
        angle_range = "unavailable"
        median_angle_step = "unavailable"
    else:
        angles_deg = np.round(np.rad2deg(np.asarray(angles_rad, dtype=np.float64)), decimals=6)
        number_of_angles = str(angles_deg.size)
        angle_range = f"{np.min(angles_deg):.8g} to {np.max(angles_deg):.8g} degrees"
        if angles_deg.size > 1:
            median_angle_step = f"{np.median(np.abs(np.diff(angles_deg))):.8g} degrees"
        else:
            median_angle_step = "unavailable"

    if projection_stack is None or projection_stack.size == 0:
        projection_shape = "unavailable"
        projection_min_max = "unavailable"
    else:
        projection_shape = str(tuple(projection_stack.shape))
        projection_min_max = f"{np.nanmin(projection_stack):.8g} / {np.nanmax(projection_stack):.8g}"
    detector_consistency_error = np.max(
        np.abs(
            np.array([detector_rows, detector_cols], dtype=np.float64)
            * np.array([d_detector_mm, d_detector_mm], dtype=np.float64)
            - np.array([detector_rows * d_detector_mm, detector_cols * d_detector_mm], dtype=np.float64)
        )
    )

    return [
        "Geometry sanity report:",
        f"Effective pixel size at object plane: {params.effective_pixel_size_um:g} um",
        f"DSO: {params.DSO_mm:g} mm",
        f"DSD: {params.DSD_mm:g} mm",
        f"Geometric magnification DSD/DSO: {magnification:.8g}",
        f"Computed TIGRE detector pixel size: {d_detector_mm:.8g} mm",
        f"Detector rows/cols: {detector_rows} x {detector_cols}",
        f"TIGRE nDetector: [{detector_rows}, {detector_cols}]",
        f"TIGRE dDetector: [{d_detector_mm:.12g}, {d_detector_mm:.12g}] mm",
        f"TIGRE sDetector: [{detector_rows * d_detector_mm:.12g}, {detector_cols * d_detector_mm:.12g}] mm",
        f"TIGRE detector consistency max abs error: {detector_consistency_error:.3g} mm",
        f"FDK filter: {fdk_filter}",
        f"Object-plane FOV: {object_fov_v_mm:.8g} mm x {object_fov_u_mm:.8g} mm",
        f"Detector-plane FOV: {detector_fov_v_mm:.8g} mm x {detector_fov_u_mm:.8g} mm",
        f"Number of angles: {number_of_angles}",
        f"Angle range: {angle_range}",
        f"Median angle step: {median_angle_step}",
        f"TIGRE angle sign multiplier: {angle_sign:+d}",
        f"Projection stack shape: {projection_shape}",
        f"TIGRE projection input min/max: {projection_min_max}",
        f"Voxel size: {params.voxel_size_um:g} um",
        f"Reconstruction volume Nx x Ny x Nz: {params.Nx} x {params.Ny} x {params.Nz}",
        f"Object offset x/y/z: {params.object_offset_x_mm:g}, {params.object_offset_y_mm:g}, {params.object_offset_z_mm:g} mm",
        f"Detector offset v/u: {off_v_mm:.8g}, {off_u_mm:.8g} mm",
        f"Center offset enabled: {params.enable_center_offset}",
        f"Center offset: {params.center_offset_px:g} px = {center_offset_mm:.8g} mm",
        f"Center offset method: {params.center_offset_method}",
        f"Center-shift sign convention: {params.center_shift_sign:+g}",
    ]


def run_fdk(
    attenuation_stack: np.ndarray,
    angles_rad: np.ndarray,
    params: GeometryParams,
    progress: Callable[[str], None] | None = None,
    extra_report_lines: list[str] | None = None,
    fdk_filter: str = "ram_lak",
    angle_sign: int = 1,
) -> ReconstructionResult:
    if attenuation_stack.ndim != 3:
        raise ValueError(f"Attenuation stack must have shape (angles, rows, cols), got {attenuation_stack.shape}")
    if attenuation_stack.shape[0] != len(angles_rad):
        raise ValueError(
            f"Number of attenuation projections ({attenuation_stack.shape[0]}) does not match angles ({len(angles_rad)})."
        )
    available, message = tigre_available()
    if not available:
        raise RuntimeError(
            "TIGRE import failed. Install TIGRE for this environment, then restart the app.\n"
            f"Original import error: {message}"
        )
    from tigre.algorithms import fdk

    if fdk_filter not in {"ram_lak", "shepp_logan", "cosine", "hamming", "hann"}:
        raise ValueError(f"FDK filter not recognized: {fdk_filter}")
    if angle_sign not in (-1, 1):
        raise ValueError("angle_sign must be +1 or -1.")

    started = datetime.now()
    detector_rows, detector_cols = attenuation_stack.shape[1:]
    geo, _ = build_tigre_geometry(params, detector_rows, detector_cols)
    projections = np.ascontiguousarray(attenuation_stack, dtype=np.float32)
    if params.enable_center_offset and params.center_offset_method == "image_shift":
        image_shift_px = params.center_shift_sign * params.center_offset_px
        if progress is not None:
            progress(f"Applying image-space horizontal center shift: {image_shift_px:g} px")
        projections = ndi_shift(
            projections,
            shift=(0.0, 0.0, float(image_shift_px)),
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float32, copy=False)

    d_detector_mm = computed_detector_pixel_mm(params)
    off_v_mm = params.detector_offset_vertical_px * d_detector_mm
    off_u_mm = params.detector_offset_horizontal_px * d_detector_mm
    center_offset_mm = 0.0
    if params.enable_center_offset and params.center_offset_method == "detector_offset":
        center_offset_mm = params.center_shift_sign * params.center_offset_px * d_detector_mm
        off_u_mm += center_offset_mm
    angles_for_tigre = angle_sign * np.asarray(angles_rad, dtype=np.float32)
    report = geometry_report(
        params,
        detector_rows,
        detector_cols,
        d_detector_mm,
        off_v_mm,
        off_u_mm,
        center_offset_mm,
        angles_rad=angles_for_tigre,
        projection_stack=projections,
        fdk_filter=fdk_filter,
        angle_sign=angle_sign,
    )
    if extra_report_lines:
        report.extend(extra_report_lines)

    if progress is not None:
        for line in report:
            progress(line)
        progress(
            "TIGRE FDK is entering native/CUDA code. Avoid Ctrl+C, the IDE stop button, or closing the terminal "
            "until it finishes; native runtimes may abort the whole process on interruption."
        )
        progress("Starting TIGRE FDK reconstruction.")
    volume = fdk(projections, geo, angles_for_tigre, filter=fdk_filter)
    finished = datetime.now()
    if progress is not None:
        progress("TIGRE FDK reconstruction finished.")
    return ReconstructionResult(
        volume=np.ascontiguousarray(volume, dtype=np.float32),
        geometry_report=report,
        started_at=started,
        finished_at=finished,
    )
