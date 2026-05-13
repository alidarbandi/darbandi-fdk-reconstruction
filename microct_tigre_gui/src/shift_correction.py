from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.ndimage import shift as ndi_shift

from .io_utils import save_image
from .metadata import ProjectionRecord


SHIFT_MODES: tuple[tuple[str, str, int, int], ...] = (
    ("A", "Mode A: row = +y_shift, col = +x_shift", 1, 1),
    ("B", "Mode B: row = -y_shift, col = +x_shift", 1, -1),
    ("C", "Mode C: row = +y_shift, col = -x_shift", -1, 1),
    ("D", "Mode D: row = -y_shift, col = -x_shift", -1, -1),
)


def apply_projection_shifts(
    stack: np.ndarray,
    x_shift_px: Sequence[float],
    y_shift_px: Sequence[float],
    x_sign: int = 1,
    y_sign: int = -1,
    interpolation_order: int = 1,
    mode: str = "nearest",
    logger: Callable[[str], None] | object | None = None,
) -> np.ndarray:
    """
    Apply per-projection x/y shifts to a projection stack.

    The stack is shaped (num_proj, rows, cols). NumPy image row coordinates
    increase downward and column coordinates increase rightward, so:
        row_shift = y_sign * y_shift_px[i]
        col_shift = x_sign * x_shift_px[i]
    """
    stack = np.asarray(stack, dtype=np.float32)
    x_shift_px = np.asarray(x_shift_px, dtype=np.float32)
    y_shift_px = np.asarray(y_shift_px, dtype=np.float32)

    if stack.ndim != 3:
        raise ValueError(f"Stack must have shape (num_proj, rows, cols), got {stack.shape}")
    n = stack.shape[0]
    if len(x_shift_px) != n or len(y_shift_px) != n:
        raise ValueError(
            "Shift arrays must match number of projections. "
            f"Stack has {n}, x has {len(x_shift_px)}, y has {len(y_shift_px)}."
        )
    if x_sign not in (-1, 1):
        raise ValueError("x_sign must be +1 or -1.")
    if y_sign not in (-1, 1):
        raise ValueError("y_sign must be +1 or -1.")
    if interpolation_order < 0:
        raise ValueError("Interpolation order must be non-negative.")

    shifted = np.empty_like(stack, dtype=np.float32)
    for index in range(n):
        row_shift = float(y_sign * y_shift_px[index])
        col_shift = float(x_sign * x_shift_px[index])
        shifted[index] = ndi_shift(
            stack[index],
            shift=(row_shift, col_shift),
            order=interpolation_order,
            mode=mode,
            prefilter=False,
        ).astype(np.float32, copy=False)
        if index < 5 or index == n - 1:
            _emit_shift_log(
                logger,
                f"Projection {index}: y_shift_px={y_shift_px[index]:.4f}, "
                f"x_shift_px={x_shift_px[index]:.4f}, applied row_shift={row_shift:.4f}, "
                f"applied col_shift={col_shift:.4f}",
            )
    return np.ascontiguousarray(shifted, dtype=np.float32)


def build_shift_sanity_report(
    shift_source_csv: str,
    x_column: str,
    y_column: str,
    x_column_found: bool,
    y_column_found: bool,
    x_shift_px: np.ndarray | None,
    y_shift_px: np.ndarray | None,
    enabled: bool,
    shift_stage: str,
    x_sign: int,
    y_sign: int,
    interpretation_sign: int,
    interpolation_order: int,
    boundary_mode: str,
    flip_vertical: bool,
    flip_horizontal: bool,
    center_offset_enabled: bool,
    center_offset_px: float,
    center_offset_mm: float,
    center_offset_method: str,
    duplicate_endpoint_removed: bool,
) -> list[str]:
    if interpretation_sign not in (-1, 1):
        raise ValueError("interpretation_sign must be +1 or -1.")
    effective_x_sign = interpretation_sign * x_sign
    effective_y_sign = interpretation_sign * y_sign
    shift_table_loaded = x_shift_px is not None and y_shift_px is not None
    interpretation_text = (
        "Apply shifts as correction values"
        if interpretation_sign == 1
        else "Apply negative of shifts as correction values"
    )

    lines = [
        "",
        "Per-projection shift correction",
        "-------------------------------",
        f"Shift table loaded: {shift_table_loaded}",
        f"Shift source CSV: {shift_source_csv or 'unavailable'}",
        f"x_shift_px column selected: {x_column or '(none)'}",
        f"y_shift_px column selected: {y_column or '(none)'}",
        f"x_shift_px column found: {x_column_found}",
        f"y_shift_px column found: {y_column_found}",
        *_shift_stats_lines("x_shift_px", x_shift_px),
        *_shift_stats_lines("y_shift_px", y_shift_px),
        f"Shift correction enabled: {enabled}",
        f"Shift application stage: {shift_stage}",
        f"Shift interpretation: {interpretation_text}",
        f"X shift sign applied to image columns: {_signed_int(effective_x_sign)}",
        f"Y shift sign applied to image rows: {_signed_int(effective_y_sign)}",
        "Applied formula:",
        f"row_shift = {_signed_int(effective_y_sign)} * y_shift_px",
        f"col_shift = {_signed_int(effective_x_sign)} * x_shift_px",
        f"Interpolation: scipy.ndimage.shift order={interpolation_order} mode={boundary_mode}",
        f"Maximum absolute applied row shift: {_max_abs_shift(y_shift_px, effective_y_sign)}",
        f"Maximum absolute applied col shift: {_max_abs_shift(x_shift_px, effective_x_sign)}",
        f"Global vertical flip before shift: {flip_vertical}",
        f"Global horizontal flip before shift: {flip_horizontal}",
        f"Center offset enabled: {center_offset_enabled}",
        f"Center offset px/mm/sign: {center_offset_px:g} px / {center_offset_mm:.8g} mm / user-controlled",
        f"Center offset method: {center_offset_method}",
        f"Duplicate endpoint removed: {duplicate_endpoint_removed}",
        "Warning:",
        "Image row coordinates increase downward. Zeiss physical Y may increase upward.",
        "If reconstructed objects smear into arcs/ovals, test the opposite Y sign.",
    ]
    if flip_vertical:
        lines.append("Vertical flip applied before per-projection shift: True. Y shift sign may need inversion.")
    if flip_horizontal:
        lines.append("Horizontal flip applied before per-projection shift: True. X shift and center-offset signs may need inversion.")
    return lines


def effective_shift_signs(x_sign: int, y_sign: int, interpretation_sign: int) -> tuple[int, int]:
    if x_sign not in (-1, 1) or y_sign not in (-1, 1) or interpretation_sign not in (-1, 1):
        raise ValueError("Shift signs must be +1 or -1.")
    return interpretation_sign * x_sign, interpretation_sign * y_sign


def save_shift_debug_previews(
    output_folder: str | Path,
    records: Sequence[ProjectionRecord],
    before_stack: np.ndarray,
    after_stack: np.ndarray,
    logger: Callable[[str], None] | None = None,
    stage_label: str = "",
) -> Path:
    if before_stack.shape != after_stack.shape:
        raise ValueError(f"Before/after stacks differ: {before_stack.shape} vs {after_stack.shape}")
    if before_stack.ndim != 3:
        raise ValueError(f"Stacks must have shape (num_proj, rows, cols), got {before_stack.shape}")
    if len(records) != before_stack.shape[0]:
        raise ValueError("Projection records must match stack length.")

    folder_name = f"shift_debug_{stage_label}" if stage_label else "shift_debug"
    folder = Path(output_folder) / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    for index in _preview_indices(before_stack.shape[0]):
        stem = Path(records[index].filename).stem
        before = np.asarray(before_stack[index], dtype=np.float32)
        after = np.asarray(after_stack[index], dtype=np.float32)
        save_image(folder / f"{stem}_before_shift.tif", before)
        save_image(folder / f"{stem}_after_shift.tif", after)
        save_image(folder / f"{stem}_after_minus_before.tif", after - before)
    if logger is not None:
        logger(f"Saved shift debug previews: {folder}")
    return folder


def _preview_indices(count: int) -> list[int]:
    candidates = [0, 99, 199, count // 2, count - 1]
    return sorted({index for index in candidates if 0 <= index < count})


def _shift_stats_lines(name: str, values: np.ndarray | None) -> list[str]:
    if values is None or len(values) == 0:
        return [f"{name} min/max/mean/std: unavailable"]
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [f"{name} min/max/mean/std: unavailable"]
    return [
        (
            f"{name} min/max/mean/std: {np.min(finite):.8g} / {np.max(finite):.8g} / "
            f"{np.mean(finite):.8g} / {np.std(finite):.8g}"
        )
    ]


def _max_abs_shift(values: np.ndarray | None, sign: int) -> str:
    if values is None or len(values) == 0:
        return "unavailable"
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "unavailable"
    return f"{np.max(np.abs(sign * finite)):.8g} px"


def _signed_int(value: int) -> str:
    return "+1" if value >= 0 else "-1"


def _emit_shift_log(logger: Callable[[str], None] | object | None, message: str) -> None:
    if logger is None:
        return
    if callable(logger):
        logger(message)
        return
    info = getattr(logger, "info", None)
    if callable(info):
        info(message)
