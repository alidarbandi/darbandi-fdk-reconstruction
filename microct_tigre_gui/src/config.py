from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
RECENT_CONFIG_POINTER = PROJECT_DIR / ".recent_config_path"


@dataclass
class AppConfig:
    projection_folder: str = ""
    reference_folder: str = ""
    metadata_csv: str = ""
    output_folder: str = ""
    filename_column: str = ""
    angle_column: str = ""
    angle_rad_column: str = ""
    x_shift_column: str = ""
    y_shift_column: str = ""
    effective_pixel_size_um: float = 11.0
    DSO_mm: float | None = None
    DSD_mm: float | None = None
    Nx: int | None = None
    Ny: int | None = None
    Nz: int | None = None
    voxel_size_um: float = 11.0
    detector_rows: int | None = None
    detector_columns: int | None = None
    object_offset_x_mm: float = 0.0
    object_offset_y_mm: float = 0.0
    object_offset_z_mm: float = 0.0
    detector_offset_vertical_px: float = 0.0
    detector_offset_horizontal_px: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    enable_center_offset: bool = False
    center_offset_px: float = 0.0
    center_offset_method: str = "detector_offset"
    center_shift_sign: float = 1.0
    epsilon: float = 1e-6
    clip_transmission: bool = True
    min_transmission: float = 1e-6
    max_transmission: float = 2.0
    max_transmission_for_log: float = 2.0
    clip_negative_attenuation_to_zero: bool = True
    remove_duplicate_endpoint_angle: bool = True
    reverse_angle_order: bool = False
    invert_angle_sign_for_tigre: bool = True
    binning_factor: int = 1
    save_transmission_stack: bool = False
    save_attenuation_stack: bool = False
    enable_projection_shift_correction: bool = True
    shift_mode_preset: str = "A"
    x_shift_sign: int = 1
    y_shift_sign: int = 1
    shift_interpretation_sign: int = 1
    shift_interpolation_order: int = 1
    shift_boundary_mode: str = "nearest"
    save_shift_debug_previews: bool = True
    shift_debug_volume_size: int = 256
    projection_shift_stage: str = "transmission"
    enable_truncation_correction: bool = False
    truncation_extension_fraction: float = 0.1
    transpose_attenuation_for_tigre: bool = False
    fdk_filter: str = "ram_lak"
    save_reconstruction_stack_folder: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AppConfig":
        allowed = {field.name for field in fields(cls)}
        clean = {key: value for key, value in values.items() if key in allowed}
        return cls(**clean)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return AppConfig.from_dict(json.load(handle))


def save_config(config: AppConfig, path: str | Path) -> Path:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    remember_recent_config(config_path)
    return config_path


def remember_recent_config(path: str | Path) -> None:
    RECENT_CONFIG_POINTER.write_text(str(Path(path).resolve()), encoding="utf-8")


def recent_config_path() -> Path | None:
    if not RECENT_CONFIG_POINTER.exists():
        return None
    text = RECENT_CONFIG_POINTER.read_text(encoding="utf-8").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None
