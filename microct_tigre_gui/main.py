from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def run_self_test() -> int:
    import numpy as np
    import pandas as pd

    from src.center_shift_search import (
        compute_metrics_for_results,
        generate_shift_values,
        metric_gradient_energy,
        metric_laplacian_variance,
        metric_shannon_entropy,
        CenterShiftPreviewResult,
    )
    from src.metadata import load_projection_geometry_table, validate_metadata
    from src.preprocessing import apply_flat_field, compute_attenuation, truncation_correction
    from src.shift_correction import apply_projection_shifts, build_shift_sanity_report
    from src.tigre_reconstruction import GeometryParams, computed_detector_pixel_mm, geometry_report, tigre_available

    raw = np.full((2, 4, 4), 10.0, dtype=np.float32)
    flat = np.full((4, 4), 20.0, dtype=np.float32)
    transmission = apply_flat_field(raw, flat, epsilon=1e-6, clip_transmission=True)
    attenuation = compute_attenuation(transmission)
    assert transmission.shape == (2, 4, 4)
    assert np.allclose(transmission, 0.5)
    assert np.allclose(attenuation, -np.log(0.5))
    assert generate_shift_values(-5.0, 5.0, 0.5) == [float(value) for value in np.arange(-5.0, 5.5, 0.5)]
    assert generate_shift_values(2.0, -2.0, 1.0) == [2.0, 1.0, 0.0, -1.0, -2.0]
    try:
        generate_shift_values(-1.0, 1.0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("generate_shift_values should reject a zero step")
    metric_image = np.zeros((16, 16), dtype=np.float32)
    metric_image[4:12, 4:12] = 1.0
    assert metric_gradient_energy(metric_image) > 0.0
    assert metric_laplacian_variance(metric_image) > 0.0
    assert metric_shannon_entropy(metric_image) >= 0.0
    metric_results = [
        CenterShiftPreviewResult(-1.0, metric_image, 0, 8, 1.0, -0.1),
        CenterShiftPreviewResult(0.0, metric_image * 0.5, 1, 8, 1.0, 0.0),
    ]
    metric_table = compute_metrics_for_results(metric_results)
    assert len(metric_table) == 2
    assert "combined_score" in metric_table[0]
    extended = truncation_correction(attenuation, extension_fraction=0.25)
    assert extended.shape == (2, 4, 6)
    params = GeometryParams(
        effective_pixel_size_um=11.0,
        DSO_mm=50.0,
        DSD_mm=100.0,
        Nx=8,
        Ny=8,
        Nz=8,
        voxel_size_um=11.0,
    )
    assert np.isclose(computed_detector_pixel_mm(params), 0.022)
    report = geometry_report(
        params,
        detector_rows=4,
        detector_cols=4,
        d_detector_mm=computed_detector_pixel_mm(params),
        off_v_mm=0.0,
        off_u_mm=0.0,
        center_offset_mm=0.0,
        angles_rad=np.deg2rad(np.array([0.0, 1.0], dtype=np.float32)),
        projection_stack=attenuation,
    )
    assert "Geometry sanity report:" in report
    assert "Effective pixel size at object plane: 11 um" in report
    assert "Number of angles: 2" in report
    assert "Median angle step: 1 degrees" in report
    assert "Projection stack shape: (2, 4, 4)" in report
    assert "FDK filter: ram_lak" in report
    assert "TIGRE angle sign multiplier: +1" in report
    inverted_report = geometry_report(
        params,
        detector_rows=4,
        detector_cols=4,
        d_detector_mm=computed_detector_pixel_mm(params),
        off_v_mm=0.0,
        off_u_mm=0.0,
        center_offset_mm=0.0,
        angles_rad=-np.deg2rad(np.array([0.0, 1.0], dtype=np.float32)),
        projection_stack=attenuation,
        angle_sign=-1,
    )
    assert "TIGRE angle sign multiplier: -1" in inverted_report
    with TemporaryDirectory() as temp_dir:
        projection_folder = Path(temp_dir)
        for filename in ("proj_000001.tif", "proj_000002.tif"):
            (projection_folder / filename).touch()
        frame = pd.DataFrame(
            {
                "tiff_file": ["proj_000001.tif", "proj_000002.tif"],
                "angle_deg": [0.0, 90.0],
                "angle_rad": [0.0, np.pi / 2.0],
                "x_shift_px": [1.0, -2.0],
                "y_shift_px": [3.0, -4.0],
            }
        )
        csv_path = projection_folder / "projection_geometry.csv"
        frame.to_csv(csv_path, index=False)
        geometry_table = load_projection_geometry_table(csv_path)
        assert geometry_table["tiff_files"] == ["proj_000001.tif", "proj_000002.tif"]
        assert np.allclose(geometry_table["x_shift_px"], np.array([1.0, -2.0], dtype=np.float32))
        validation = validate_metadata(
            frame,
            projection_folder,
            "tiff_file",
            "angle_deg",
            "angle_rad",
            "x_shift_px",
            "y_shift_px",
        )
        assert validation.angle_input_source == "radian column 'angle_rad'"
        assert np.allclose(validation.angles_rad, np.array([0.0, np.pi / 2.0], dtype=np.float32))
        assert np.allclose(validation.x_shift_px, np.array([1.0, -2.0], dtype=np.float32))
        assert np.allclose(validation.y_shift_px, np.array([3.0, -4.0], dtype=np.float32))
        assert validation.table_rows[0]["angle_rad"] == "0"
        assert validation.table_rows[0]["x_shift_px"] == "1"

        fallback_frame = frame.copy()
        fallback_frame["angle_rad"] = ""
        fallback = validate_metadata(
            fallback_frame,
            projection_folder,
            "tiff_file",
            "angle_deg",
            "angle_rad",
            "x_shift_px",
            "y_shift_px",
        )
        assert fallback.angle_input_source == "degree column 'angle_deg' converted to radians"
        assert np.allclose(fallback.angles_rad, np.deg2rad(np.array([0.0, 90.0], dtype=np.float32)))
        assert any("falling back to degree column" in warning for warning in fallback.warnings)
    test_stack = np.zeros((1, 5, 5), dtype=np.float32)
    test_stack[0, 2, 2] = 1.0
    shifted = apply_projection_shifts(test_stack, [1.0], [1.0], x_sign=1, y_sign=-1, interpolation_order=0)
    assert shifted[0, 1, 3] == 1.0
    rectangular = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    transposed = np.transpose(rectangular, (0, 2, 1))
    assert transposed.shape == (2, 4, 3)
    assert transposed[1, 3, 2] == rectangular[1, 2, 3]
    shift_report = build_shift_sanity_report(
        shift_source_csv="projection_geometry.csv",
        x_column="x_shift_px",
        y_column="y_shift_px",
        x_column_found=True,
        y_column_found=True,
        x_shift_px=np.array([1.0], dtype=np.float32),
        y_shift_px=np.array([1.0], dtype=np.float32),
        enabled=True,
        shift_stage="transmission",
        x_sign=1,
        y_sign=-1,
        interpretation_sign=1,
        interpolation_order=1,
        boundary_mode="nearest",
        flip_vertical=False,
        flip_horizontal=False,
        center_offset_enabled=False,
        center_offset_px=0.0,
        center_offset_mm=0.0,
        center_offset_method="detector_offset",
        duplicate_endpoint_removed=False,
    )
    assert "row_shift = -1 * y_shift_px" in shift_report
    assert "col_shift = +1 * x_shift_px" in shift_report
    available, message = tigre_available()
    print("Self-test passed.")
    print(f"TIGRE available: {available} ({message})")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MicroCT TIGRE FDK reconstruction GUI")
    parser.add_argument("--self-test", action="store_true", help="Run lightweight non-GUI checks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.self_test:
        return run_self_test()
    try:
        from src.gui import run_app
    except ImportError as exc:
        print(
            "Could not import GUI dependencies. Install requirements with:\n"
            "python -m pip install -r requirements.txt\n\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 2
    return run_app(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
