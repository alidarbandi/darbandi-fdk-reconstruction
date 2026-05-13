from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import tifffile

from .metadata import ProjectionRecord


TIFF_PATTERNS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")


def list_tiff_files(folder: str | Path) -> list[Path]:
    path = Path(folder)
    files: list[Path] = []
    for pattern in TIFF_PATTERNS:
        files.extend(path.glob(pattern))
    return sorted(set(files))


def read_tiff_float32(path: str | Path, binning: int = 1) -> np.ndarray:
    array = tifffile.imread(str(path)).astype(np.float32, copy=False)
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D TIFF image, got shape {array.shape} for {path}")
    if binning > 1:
        array = block_average_2d(array, binning)
    return np.ascontiguousarray(array, dtype=np.float32)


def block_average_2d(array: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return array.astype(np.float32, copy=False)
    if factor not in (2, 4):
        raise ValueError("Binning factor must be 1, 2, or 4.")
    rows, cols = array.shape
    trim_rows = rows - rows % factor
    trim_cols = cols - cols % factor
    if trim_rows <= 0 or trim_cols <= 0:
        raise ValueError(f"Image shape {array.shape} is too small for {factor}x binning.")
    trimmed = array[:trim_rows, :trim_cols].astype(np.float32, copy=False)
    return trimmed.reshape(trim_rows // factor, factor, trim_cols // factor, factor).mean(axis=(1, 3))


def estimate_stack_memory_gb(num_images: int, rows: int, cols: int, bytes_per_pixel: int = 4) -> float:
    return num_images * rows * cols * bytes_per_pixel / 1024**3


def load_projection_stack(
    records: Sequence[ProjectionRecord],
    binning: int = 1,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    if not records:
        raise ValueError("No projection records are available.")
    first = read_tiff_float32(records[0].path, binning=binning)
    rows, cols = first.shape
    stack = np.empty((len(records), rows, cols), dtype=np.float32)
    stack[0] = first
    if progress is not None:
        progress(f"Loaded projection 1/{len(records)}: {records[0].filename}")
    for index, record in enumerate(records[1:], start=1):
        if not record.path.exists():
            raise FileNotFoundError(f"Projection listed in metadata is missing: {record.filename}")
        image = read_tiff_float32(record.path, binning=binning)
        if image.shape != (rows, cols):
            raise ValueError(
                f"Projection shape mismatch for {record.filename}: {image.shape}, expected {(rows, cols)}"
            )
        stack[index] = image
        if progress is not None and (index + 1 == len(records) or (index + 1) % 25 == 0):
            progress(f"Loaded projection {index + 1}/{len(records)}: {record.filename}")
    return stack


def save_image(path: str | Path, image: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output), np.asarray(image, dtype=np.float32), photometric="minisblack")
    return output


def save_png_image(path: str | Path, image: np.ndarray) -> Path:
    from matplotlib import image as mpimg

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size:
        low, high = np.percentile(finite, [1, 99])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(finite))
            high = float(np.max(finite))
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    mpimg.imsave(str(output), scaled, cmap="gray", vmin=0.0, vmax=1.0)
    return output


def save_stack_tiff(path: str | Path, stack: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output), np.asarray(stack, dtype=np.float32), photometric="minisblack")
    return output


def save_stack_folder(
    folder: str | Path,
    stack: np.ndarray,
    prefix: str,
    progress: Callable[[str], None] | None = None,
) -> Path:
    output = Path(folder)
    output.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(stack):
        path = output / f"{prefix}_{index:06d}.tif"
        tifffile.imwrite(str(path), np.asarray(image, dtype=np.float32), photometric="minisblack")
        if progress is not None and (index + 1 == stack.shape[0] or (index + 1) % 25 == 0):
            progress(f"Saved {index + 1}/{stack.shape[0]} slices to {output}")
    return output
