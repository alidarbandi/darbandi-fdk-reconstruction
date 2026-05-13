from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .io_utils import list_tiff_files, read_tiff_float32


def compute_average_flat_field(
    reference_folder: str | Path,
    expected_raw_shape: tuple[int, int] | None = None,
    binning: int = 1,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, int]:
    files = list_tiff_files(reference_folder)
    if not files:
        raise FileNotFoundError(f"No TIFF reference images found in {reference_folder}")

    accumulator: np.ndarray | None = None
    binned_shape: tuple[int, int] | None = None
    for index, path in enumerate(files, start=1):
        raw = read_tiff_float32(path, binning=1)
        if expected_raw_shape is not None and raw.shape != expected_raw_shape:
            raise ValueError(
                f"Reference shape mismatch for {path.name}: {raw.shape}, expected {expected_raw_shape}"
            )
        image = read_tiff_float32(path, binning=binning)
        if accumulator is None:
            binned_shape = image.shape
            accumulator = np.zeros_like(image, dtype=np.float64)
        if image.shape != binned_shape:
            raise ValueError(f"Reference shape mismatch for {path.name}: {image.shape}, expected {binned_shape}")
        accumulator += image.astype(np.float64, copy=False)
        if progress is not None and (index == len(files) or index % 10 == 0):
            progress(f"Averaged reference {index}/{len(files)}: {path.name}")

    assert accumulator is not None
    flat = (accumulator / len(files)).astype(np.float32)
    return flat, len(files)


def apply_orientation_to_image(image: np.ndarray, flip_horizontal: bool, flip_vertical: bool) -> np.ndarray:
    oriented = image
    if flip_vertical:
        oriented = np.flip(oriented, axis=0)
    if flip_horizontal:
        oriented = np.flip(oriented, axis=1)
    return np.ascontiguousarray(oriented, dtype=np.float32)


def apply_orientation_to_stack(stack: np.ndarray, flip_horizontal: bool, flip_vertical: bool) -> np.ndarray:
    oriented = stack
    if flip_vertical:
        oriented = np.flip(oriented, axis=1)
    if flip_horizontal:
        oriented = np.flip(oriented, axis=2)
    return np.ascontiguousarray(oriented, dtype=np.float32)


def apply_flat_field(
    projection_stack: np.ndarray,
    flat_field: np.ndarray,
    epsilon: float = 1e-6,
    clip_transmission: bool = True,
    min_transmission: float = 1e-6,
    max_transmission: float = 2.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> np.ndarray:
    if projection_stack.ndim != 3:
        raise ValueError(f"Projection stack must have shape (angles, rows, cols), got {projection_stack.shape}")
    if flat_field.shape != projection_stack.shape[1:]:
        raise ValueError(f"Flat-field shape {flat_field.shape} does not match projections {projection_stack.shape[1:]}")
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    if clip_transmission and min_transmission <= 0:
        raise ValueError("Minimum transmission must be positive when clipping is enabled.")

    # There is no dark-current image for this workflow, so the physically intended
    # correction is T = I/F, not T = (I-D)/(F-D).
    oriented_stack = apply_orientation_to_stack(projection_stack, flip_horizontal, flip_vertical)
    oriented_flat = apply_orientation_to_image(flat_field, flip_horizontal, flip_vertical)
    flat_safe = np.maximum(oriented_flat, np.float32(epsilon))
    transmission = oriented_stack / flat_safe[None, :, :]
    if clip_transmission:
        transmission = np.clip(transmission, min_transmission, max_transmission)
    return np.ascontiguousarray(transmission, dtype=np.float32)


def compute_attenuation(
    transmission: np.ndarray,
    epsilon: float = 1e-6,
    max_transmission_for_log: float = 2.0,
    clip_negative_to_zero: bool = False,
) -> np.ndarray:
    if transmission.ndim != 3:
        raise ValueError(f"Transmission stack must have shape (angles, rows, cols), got {transmission.shape}")
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    if max_transmission_for_log <= epsilon:
        raise ValueError("Maximum transmission for log must be larger than epsilon.")

    # TIGRE FDK must receive attenuation line integrals p = -ln(I/F), not raw
    # 16-bit intensity images and not normalized transmission images.
    transmission_safe = np.clip(transmission, epsilon, max_transmission_for_log)
    attenuation = -np.log(transmission_safe).astype(np.float32)
    if clip_negative_to_zero:
        attenuation = np.maximum(attenuation, 0.0)
    return np.ascontiguousarray(attenuation, dtype=np.float32)


def truncation_correction(projections: np.ndarray, extension_fraction: float = 0.1) -> np.ndarray:
    if projections.ndim != 3:
        raise ValueError(f"Projection stack must have shape (angles, rows, cols), got {projections.shape}")
    if extension_fraction <= 0:
        raise ValueError("Truncation correction extension fraction must be positive.")

    num_images, rows, cols = projections.shape
    n_ext = int(cols * extension_fraction)
    if n_ext <= 0:
        raise ValueError(
            f"Truncation correction extension fraction {extension_fraction:g} is too small for {cols} columns."
        )
    extended = np.pad(
        np.asarray(projections, dtype=np.float32),
        pad_width=((0, 0), (0, 0), (n_ext, n_ext)),
        mode="symmetric",
    )
    ramp = np.linspace(0.0, 1.0, n_ext, endpoint=True, dtype=np.float32)
    extended[:, :, :n_ext] *= ramp[None, None, :]
    extended[:, :, -n_ext:] *= ramp[::-1][None, None, :]
    return np.ascontiguousarray(extended, dtype=np.float32)
