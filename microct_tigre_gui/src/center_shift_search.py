from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable

import numpy as np

from .tigre_reconstruction import GeometryParams, computed_detector_pixel_mm, run_fdk


MAX_PREVIEW_SOFT_LIMIT = 51
MAX_PREVIEW_DEFAULT_LIMIT = 101
METRIC_EPSILON = 1e-12


@dataclass
class CenterShiftPreviewResult:
    shift_px: float
    preview_image: np.ndarray
    preview_index: int
    slice_index: int
    center_shift_sign: float
    detector_offset_physical: float
    metrics: dict[str, float] | None = None
    reconstruction_time_s: float | None = None


def generate_shift_values(start_px: float, end_px: float, step_px: float) -> list[float]:
    if not np.isfinite(start_px) or not np.isfinite(end_px) or not np.isfinite(step_px):
        raise ValueError("Start shift, end shift, and step must be finite numbers.")
    if step_px <= 0:
        raise ValueError("Center-offset search step must be greater than zero.")

    direction = 1.0 if end_px >= start_px else -1.0
    signed_step = direction * abs(step_px)
    tolerance = max(abs(step_px) * 1e-7, 1e-9)
    values: list[float] = []
    current = float(start_px)

    while direction * (current - end_px) <= tolerance:
        values.append(round(current, 10))
        current += signed_step
        if len(values) > 100000:
            raise ValueError("Center-offset search generated too many values.")

    if values and abs(values[-1] - end_px) <= tolerance:
        values[-1] = round(float(end_px), 10)
    elif not values:
        values.append(round(float(start_px), 10))
    return values


def make_geometry_with_center_shift(base_geo, shift_px: float, pixel_size_u: float, center_shift_sign: float = 1.0):
    """
    Return a copied TIGRE geometry object with horizontal detector offset adjusted.

    TIGRE stores detector offset as [vertical, horizontal] in this app.
    """
    if pixel_size_u is None or not np.isfinite(pixel_size_u) or pixel_size_u <= 0:
        raise ValueError("Detector pixel size must be a positive finite value.")
    if center_shift_sign not in (-1, 1, -1.0, 1.0):
        raise ValueError("Center-shift sign convention must be +1 or -1.")

    geo_shifted = deepcopy(base_geo)
    detector_offset = np.asarray(getattr(geo_shifted, "offDetector", np.zeros(2)), dtype=np.float64).copy()
    if detector_offset.ndim == 1:
        if detector_offset.size < 2:
            detector_offset = np.resize(detector_offset, 2).astype(np.float64)
        detector_offset[1] += float(center_shift_sign) * float(shift_px) * float(pixel_size_u)
    elif detector_offset.ndim == 2:
        detector_offset[1, ...] += float(center_shift_sign) * float(shift_px) * float(pixel_size_u)
    else:
        raise ValueError(f"Unsupported offDetector shape: {detector_offset.shape}")
    geo_shifted.offDetector = detector_offset
    return geo_shifted


def reconstruct_center_shift_preview(
    projections: np.ndarray,
    angles: np.ndarray,
    base_params: GeometryParams,
    shift_px: float,
    preview_slice_index: int,
    preview_mode: str = "single_slice",
    band_thickness: int = 1,
    center_shift_sign: float = 1.0,
    progress: Callable[[str], None] | None = None,
    fdk_filter: str = "ram_lak",
    angle_sign: int = 1,
) -> CenterShiftPreviewResult:
    if projections.ndim != 3:
        raise ValueError(f"Projection stack must be 3-D, got shape {projections.shape}.")
    if center_shift_sign not in (-1, 1, -1.0, 1.0):
        raise ValueError("Center-shift sign convention must be +1 or -1.")
    band_nz = _preview_band_size(preview_mode, band_thickness)
    d_detector_mm = computed_detector_pixel_mm(base_params)
    detector_offset_physical = float(center_shift_sign) * float(shift_px) * d_detector_mm
    preview_params = replace(
        base_params,
        Nz=band_nz,
        enable_center_offset=True,
        center_offset_px=float(shift_px),
        center_offset_method="detector_offset",
        center_shift_sign=float(center_shift_sign),
    )

    started = perf_counter()
    result = run_fdk(
        projections,
        angles,
        preview_params,
        progress=progress,
        fdk_filter=fdk_filter,
        angle_sign=angle_sign,
        extra_report_lines=[
            "",
            "Center offset preview",
            "---------------------",
            f"Requested full-volume preview slice index: {preview_slice_index}",
            "Preview reconstruction uses a centered axial slab in this implementation.",
            f"Preview mode: {preview_mode}",
            f"Preview slab thickness: {band_nz}",
            f"Center-shift sign convention: {float(center_shift_sign):+g}",
            f"Candidate shift: {float(shift_px):.4f} px",
            f"TIGRE detector offset correction: {detector_offset_physical:.8g} mm",
        ],
    )
    elapsed = perf_counter() - started
    volume = np.asarray(result.volume, dtype=np.float32)
    if volume.ndim != 3 or volume.shape[0] == 0:
        raise RuntimeError(f"Preview reconstruction returned unexpected shape: {volume.shape}")
    image = np.ascontiguousarray(volume[volume.shape[0] // 2], dtype=np.float32)
    return CenterShiftPreviewResult(
        shift_px=float(shift_px),
        preview_image=image,
        preview_index=0,
        slice_index=int(preview_slice_index),
        center_shift_sign=float(center_shift_sign),
        detector_offset_physical=detector_offset_physical,
        reconstruction_time_s=elapsed,
    )


def _preview_band_size(preview_mode: str, band_thickness: int) -> int:
    if preview_mode == "thin_band":
        return max(1, int(band_thickness))
    return 1


def prepare_image_for_metrics(
    image: np.ndarray,
    crop_fraction: float = 0.1,
    percentile_clip: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Metric image must be 2-D, got shape {array.shape}.")

    crop_fraction = min(max(float(crop_fraction), 0.0), 0.45)
    rows, cols = array.shape
    row_crop = int(rows * crop_fraction)
    col_crop = int(cols * crop_fraction)
    if row_crop > 0 or col_crop > 0:
        array = array[row_crop : rows - row_crop or rows, col_crop : cols - col_crop or cols]

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros((1, 1), dtype=np.float64)
    low, high = np.percentile(finite, percentile_clip)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros_like(array, dtype=np.float64)

    clipped = np.clip(np.nan_to_num(array, nan=low, posinf=high, neginf=low), low, high)
    return (clipped - low) / (high - low + METRIC_EPSILON)


def metric_gradient_energy(image: np.ndarray) -> float:
    prepared = prepare_image_for_metrics(image)
    if prepared.shape[0] < 2 or prepared.shape[1] < 2:
        return 0.0
    gy, gx = np.gradient(prepared)
    value = float(np.mean(gx * gx + gy * gy))
    return value if np.isfinite(value) else 0.0


def metric_laplacian_variance(image: np.ndarray) -> float:
    prepared = prepare_image_for_metrics(image)
    laplacian = np.zeros_like(prepared)
    laplacian[1:-1, 1:-1] = (
        prepared[:-2, 1:-1]
        + prepared[2:, 1:-1]
        + prepared[1:-1, :-2]
        + prepared[1:-1, 2:]
        - 4.0 * prepared[1:-1, 1:-1]
    )
    value = float(np.var(laplacian))
    return value if np.isfinite(value) else 0.0


def metric_shannon_entropy(image: np.ndarray, bins: int = 256) -> float:
    prepared = prepare_image_for_metrics(image)
    hist, _ = np.histogram(prepared[np.isfinite(prepared)], bins=bins, range=(0.0, 1.0), density=False)
    total = np.sum(hist)
    if total <= 0:
        return 0.0
    probabilities = hist.astype(np.float64) / float(total)
    probabilities = probabilities[probabilities > 0]
    value = float(-np.sum(probabilities * np.log(probabilities + METRIC_EPSILON)))
    return max(value, 0.0) if np.isfinite(value) else 0.0


def compute_metrics_for_results(results: list[CenterShiftPreviewResult]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for result in results:
        metrics = {
            "shift_px": float(result.shift_px),
            "gradient_energy": metric_gradient_energy(result.preview_image),
            "laplacian_variance": metric_laplacian_variance(result.preview_image),
            "entropy": metric_shannon_entropy(result.preview_image),
        }
        rows.append(metrics)

    gradient_scores = _normalize_metric([row["gradient_energy"] for row in rows], higher_is_better=True)
    laplacian_scores = _normalize_metric([row["laplacian_variance"] for row in rows], higher_is_better=True)
    entropy_scores = _normalize_metric([row["entropy"] for row in rows], higher_is_better=False)
    for index, row in enumerate(rows):
        row["combined_score"] = (
            0.4 * gradient_scores[index] + 0.4 * laplacian_scores[index] + 0.2 * entropy_scores[index]
        )
    for index, row in enumerate(rows):
        row["gradient_rank"] = float(_rank_index(rows, "gradient_energy", index, higher_is_better=True))
        row["laplacian_rank"] = float(_rank_index(rows, "laplacian_variance", index, higher_is_better=True))
        row["entropy_rank"] = float(_rank_index(rows, "entropy", index, higher_is_better=False))
        row["combined_rank"] = float(_rank_index(rows, "combined_score", index, higher_is_better=True))
        results[index].metrics = {
            key: value for key, value in row.items() if key != "shift_px"
        }
    return rows


def recommend_metric_index(metric_table: list[dict[str, float]], metric_key: str) -> int | None:
    if not metric_table:
        return None
    values = np.asarray([row.get(metric_key, np.nan) for row in metric_table], dtype=np.float64)
    if not np.any(np.isfinite(values)):
        return None
    if metric_key == "entropy":
        return int(np.nanargmin(values))
    return int(np.nanargmax(values))


def metric_display_name(metric_key: str) -> str:
    return {
        "combined_score": "Combined score",
        "gradient_energy": "Gradient energy",
        "laplacian_variance": "Laplacian variance",
        "entropy": "Entropy",
    }.get(metric_key, metric_key)


def _normalize_metric(values: list[float], higher_is_better: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float64)
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    normalized = (np.nan_to_num(array, nan=min_value) - min_value) / (max_value - min_value + METRIC_EPSILON)
    if higher_is_better:
        return normalized
    return 1.0 - normalized


def _rank_index(rows: list[dict[str, float]], key: str, index: int, higher_is_better: bool) -> int:
    decorated = [(row.get(key, np.nan), row_index) for row_index, row in enumerate(rows)]
    decorated = [(value, row_index) for value, row_index in decorated if np.isfinite(value)]
    decorated.sort(key=lambda item: item[0], reverse=higher_is_better)
    for rank, (_, row_index) in enumerate(decorated, start=1):
        if row_index == index:
            return rank
    return len(rows)
