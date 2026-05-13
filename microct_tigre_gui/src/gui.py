from __future__ import annotations

import json
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .qt_compat import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QObject,
    QThread,
    QTimer,
    QT_BACKEND,
    Qt,
    Signal,
    Slot,
    exec_app,
)

from .center_shift_search import (
    MAX_PREVIEW_DEFAULT_LIMIT,
    MAX_PREVIEW_SOFT_LIMIT,
    CenterShiftPreviewResult,
    compute_metrics_for_results,
    generate_shift_values,
    metric_display_name,
    recommend_metric_index,
    reconstruct_center_shift_preview,
)
from .config import AppConfig, load_config, recent_config_path, save_config
from .io_utils import (
    estimate_stack_memory_gb,
    load_projection_stack,
    read_tiff_float32,
    save_image,
    save_png_image,
    save_stack_folder,
    save_stack_tiff,
)
from .logging_utils import timestamp, write_processing_log
from .metadata import (
    ANGLE_HINTS,
    ANGLE_RAD_HINTS,
    FILENAME_HINTS,
    MetadataValidation,
    X_SHIFT_HINTS,
    Y_SHIFT_HINTS,
    find_metadata_csv,
    load_metadata_csv,
    suggest_column,
    validate_metadata,
)
from .plotting import draw_center_shift_metric_plot, draw_graph, draw_image, make_canvas
from .preprocessing import apply_flat_field, compute_attenuation, compute_average_flat_field, truncation_correction
from .shift_correction import (
    SHIFT_MODES,
    apply_projection_shifts,
    build_shift_sanity_report,
    effective_shift_signs,
    save_shift_debug_previews,
)
from .tigre_reconstruction import GeometryParams, computed_detector_pixel_mm, run_fdk, tigre_available


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
COMBO_DOWN_ARROW_PATH = (ASSETS_DIR / "combo_down_arrow.xpm").as_posix()
SPIN_UP_ARROW_PATH = (ASSETS_DIR / "spin_up_arrow.xpm").as_posix()
SPIN_DOWN_ARROW_PATH = (ASSETS_DIR / "spin_down_arrow.xpm").as_posix()
SPLITTER_VERTICAL_GRIP_PATH = (ASSETS_DIR / "splitter_vertical_grip.xpm").as_posix()
SCROLLBAR_VERTICAL_GRIP_PATH = (ASSETS_DIR / "scrollbar_vertical_grip.xpm").as_posix()
SCROLLBAR_HORIZONTAL_GRIP_PATH = (ASSETS_DIR / "scrollbar_horizontal_grip.xpm").as_posix()

APP_STYLE_SHEET = """
QWidget {
    background-color: #273A59;
    color: #F4F8FF;
    font-size: 10pt;
    selection-background-color: #78A6E6;
    selection-color: #102033;
}

QMainWindow,
QScrollArea,
QSplitter,
QTabWidget::pane {
    background-color: #273A59;
}

QGroupBox {
    background-color: #2E4568;
    border: 1px solid #5F78A0;
    border-radius: 6px;
    margin-top: 18px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: #FFFFFF;
    background-color: #273A59;
}

QLabel,
QCheckBox,
QRadioButton {
    background: transparent;
    color: #F4F8FF;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #A8C7F7;
    border-radius: 3px;
    background-color: #1F2F49;
}

QCheckBox::indicator:checked {
    background-color: #78A6E6;
    border-color: #D8E8FF;
}

QSplitter::handle {
    background-color: #78A6E6;
    border: 1px solid #D8E8FF;
}

QSplitter::handle:horizontal {
    image: url("__SPLITTER_VERTICAL_GRIP_PATH__");
    width: 12px;
    margin: 0 2px;
}

QSplitter::handle:horizontal:hover {
    background-color: #A8C7F7;
    border-color: #FFFFFF;
}

QSplitter::handle:horizontal:pressed {
    background-color: #3F5F8B;
}

QLineEdit,
QTextEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QTableWidget {
    background-color: #1F2F49;
    color: #F8FBFF;
    border: 1px solid #6C86AE;
    border-radius: 4px;
    padding: 4px 6px;
}

QComboBox {
    padding-right: 34px;
}

QSpinBox,
QDoubleSpinBox {
    padding-right: 30px;
}

QLineEdit:focus,
QTextEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {
    border: 1px solid #A8C7F7;
}

QLineEdit:read-only,
QSpinBox:read-only,
QDoubleSpinBox:read-only {
    background-color: #263954;
    color: #DDE8F7;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    background-color: #3F5F8B;
    border-left: 1px solid #8EAEE0;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
    width: 30px;
}

QComboBox::drop-down:hover {
    background-color: #4C70A2;
    border-left-color: #B8D1FA;
}

QComboBox::drop-down:pressed {
    background-color: #213451;
}

QComboBox::down-arrow {
    image: url("__COMBO_DOWN_ARROW_PATH__");
    width: 16px;
    height: 16px;
}

QComboBox::down-arrow:on {
    top: 1px;
}

QSpinBox::up-button,
QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    background-color: #3F5F8B;
    border-left: 1px solid #8EAEE0;
    border-bottom: 1px solid #8EAEE0;
    border-top-right-radius: 4px;
    width: 26px;
    height: 15px;
}

QSpinBox::down-button,
QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    background-color: #3F5F8B;
    border-left: 1px solid #8EAEE0;
    border-top: 1px solid #8EAEE0;
    border-bottom-right-radius: 4px;
    width: 26px;
    height: 15px;
}

QSpinBox::up-button:hover,
QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover {
    background-color: #4C70A2;
    border-color: #B8D1FA;
}

QSpinBox::up-button:pressed,
QSpinBox::down-button:pressed,
QDoubleSpinBox::up-button:pressed,
QDoubleSpinBox::down-button:pressed {
    background-color: #213451;
}

QSpinBox::up-button:disabled,
QSpinBox::down-button:disabled,
QDoubleSpinBox::up-button:disabled,
QDoubleSpinBox::down-button:disabled {
    background-color: #33445F;
    border-color: #52657F;
}

QSpinBox::up-arrow,
QDoubleSpinBox::up-arrow {
    image: url("__SPIN_UP_ARROW_PATH__");
    width: 16px;
    height: 10px;
}

QSpinBox::down-arrow,
QDoubleSpinBox::down-arrow {
    image: url("__SPIN_DOWN_ARROW_PATH__");
    width: 16px;
    height: 10px;
}

QComboBox QAbstractItemView {
    background-color: #1F2F49;
    color: #F8FBFF;
    selection-background-color: #78A6E6;
    selection-color: #102033;
    border: 1px solid #6C86AE;
    outline: 0;
}

QPushButton {
    background-color: #3F5F8B;
    color: #FFFFFF;
    border: 1px solid #8EAEE0;
    border-radius: 5px;
    padding: 7px 10px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #4C70A2;
    border-color: #B8D1FA;
}

QPushButton:pressed {
    background-color: #213451;
}

QPushButton:disabled,
QLineEdit:disabled,
QTextEdit:disabled,
QComboBox:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled,
QCheckBox:disabled,
QLabel:disabled {
    background-color: #33445F;
    color: #B8C5D8;
    border-color: #52657F;
}

QTabBar::tab {
    background-color: #213451;
    color: #DDE8F7;
    border: 1px solid #5F78A0;
    border-bottom: none;
    padding: 8px 12px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #2E4568;
    color: #FFFFFF;
}

QHeaderView::section {
    background-color: #354F75;
    color: #FFFFFF;
    border: 1px solid #6C86AE;
    padding: 5px;
}

QTableWidget {
    gridline-color: #5F78A0;
    alternate-background-color: #263954;
}

QTableWidget::item {
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #78A6E6;
    color: #102033;
}

QScrollBar:vertical {
    background-color: #18263E;
    border-left: 1px solid #D8E8FF;
    width: 18px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #78A6E6;
    image: url("__SCROLLBAR_VERTICAL_GRIP_PATH__");
    border: 1px solid #FFFFFF;
    border-radius: 7px;
    min-height: 72px;
    margin: 3px 2px;
}

QScrollBar::handle:vertical:hover {
    background-color: #A8C7F7;
    border-color: #FFFFFF;
}

QScrollBar::handle:vertical:pressed {
    background-color: #3F5F8B;
}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background-color: #18263E;
}

QScrollBar:horizontal {
    background-color: #18263E;
    border-top: 1px solid #D8E8FF;
    height: 18px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #78A6E6;
    image: url("__SCROLLBAR_HORIZONTAL_GRIP_PATH__");
    border: 1px solid #FFFFFF;
    border-radius: 7px;
    min-width: 72px;
    margin: 2px 3px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #A8C7F7;
    border-color: #FFFFFF;
}

QScrollBar::handle:horizontal:pressed {
    background-color: #3F5F8B;
}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background-color: #18263E;
}

QScrollBar::add-line,
QScrollBar::sub-line {
    width: 0;
    height: 0;
}

QSlider::groove:horizontal {
    background-color: #1F2F49;
    border: 1px solid #6C86AE;
    border-radius: 4px;
    height: 8px;
}

QSlider::handle:horizontal {
    background-color: #78A6E6;
    border: 1px solid #D8E8FF;
    border-radius: 6px;
    width: 16px;
    margin: -5px 0;
}

QSlider::handle:horizontal:hover {
    background-color: #A8C7F7;
}

QStatusBar {
    background-color: #1F2F49;
    color: #F4F8FF;
    border-top: 1px solid #5F78A0;
}
""".replace("__COMBO_DOWN_ARROW_PATH__", COMBO_DOWN_ARROW_PATH).replace(
    "__SPIN_UP_ARROW_PATH__", SPIN_UP_ARROW_PATH
).replace("__SPIN_DOWN_ARROW_PATH__", SPIN_DOWN_ARROW_PATH).replace(
    "__SPLITTER_VERTICAL_GRIP_PATH__", SPLITTER_VERTICAL_GRIP_PATH
).replace("__SCROLLBAR_VERTICAL_GRIP_PATH__", SCROLLBAR_VERTICAL_GRIP_PATH).replace(
    "__SCROLLBAR_HORIZONTAL_GRIP_PATH__", SCROLLBAR_HORIZONTAL_GRIP_PATH
)


class Worker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, fn: Callable[[Callable[[str], None]], Any]) -> None:
        super().__init__()
        self.fn = fn

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(self.log.emit)
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.finished.emit(result)


class MicroCTReconstructionWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Darbandi FDK reconsturction")
        self.resize(1480, 900)
        self.setMinimumSize(1180, 720)

        self.metadata_frame = None
        self.validation: MetadataValidation | None = None
        self.expected_raw_shape: tuple[int, int] | None = None
        self.flat_field: np.ndarray | None = None
        self.raw_stack: np.ndarray | None = None
        self.transmission_stack: np.ndarray | None = None
        self.attenuation_stack: np.ndarray | None = None
        self.attenuation_preprocessing_report: list[str] = []
        self.reconstruction: np.ndarray | None = None
        self.log_lines: list[str] = []
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self.center_shift_results: list[CenterShiftPreviewResult] = []
        self.current_center_shift_preview_index = 0
        self.auto_best_shift_px: float | None = None
        self.auto_best_preview_index: int | None = None
        self.auto_metric_table: list[dict[str, float]] | None = None
        self.center_shift_preview_limits: tuple[float, float] | None = None
        self.image_zoom_factor = 1.0

        self._build_ui()
        self._connect_signals()
        self._sync_center_shift_preview_slice_range()
        self._update_center_shift_preview_controls()
        self._update_image_zoom_label()
        self._log(f"Application started with Qt backend: {QT_BACKEND}.")
        available, message = tigre_available()
        self._log(message if available else f"TIGRE warning: {message}")
        if not available:
            self.statusBar().showMessage("TIGRE is not importable; preprocessing still works.")
        QTimer.singleShot(0, self._maybe_offer_recent_config)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(12)
        self.setCentralWidget(splitter)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        self._build_path_group(left_layout)
        self._build_metadata_group(left_layout)
        self._build_preprocessing_group(left_layout)
        self._build_geometry_group(left_layout)
        self._build_orientation_group(left_layout)
        self._build_center_offset_search_group(left_layout)
        self._build_shift_group(left_layout)
        self._build_reconstruction_group(left_layout)
        self._build_config_group(left_layout)
        left_layout.addStretch(1)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(10, 10, 10, 10)
        self._build_display_group(center_layout)
        splitter.addWidget(center)

        right = QTabWidget()
        self._build_right_tabs(right)
        splitter.addWidget(right)
        splitter.setSizes([420, 690, 420])

    def _build_path_group(self, parent: QVBoxLayout) -> None:
        group = QGroupBox("Data paths")
        layout = QGridLayout(group)
        self.projection_folder_edit = QLineEdit()
        self.reference_folder_edit = QLineEdit()
        self.metadata_path_edit = QLineEdit()
        self.output_folder_edit = QLineEdit()
        rows = [
            ("Projection folder", self.projection_folder_edit, self._browse_projection_folder),
            ("Reference / flat-field folder", self.reference_folder_edit, self._browse_reference_folder),
            ("Metadata CSV or folder", self.metadata_path_edit, self._browse_metadata_file),
            ("Output folder", self.output_folder_edit, self._browse_output_folder),
        ]
        for row, (label, edit, browse) in enumerate(rows):
            layout.addWidget(QLabel(label), row, 0)
            layout.addWidget(edit, row, 1)
            button = QPushButton("Browse")
            button.clicked.connect(browse)
            layout.addWidget(button, row, 2)
        metadata_folder_button = QPushButton("Folder")
        metadata_folder_button.clicked.connect(self._browse_metadata_folder)
        layout.addWidget(metadata_folder_button, 2, 3)
        parent.addWidget(group)

    def _build_metadata_group(self, parent: QVBoxLayout) -> None:
        group = QGroupBox("Metadata")
        layout = QGridLayout(group)
        self.filename_column_combo = QComboBox()
        self.angle_column_combo = QComboBox()
        self.angle_rad_column_combo = QComboBox()
        self.x_shift_column_combo = QComboBox()
        self.y_shift_column_combo = QComboBox()
        self.remove_duplicate_endpoint_check = QCheckBox("Remove duplicate 0/360 endpoint")
        self.remove_duplicate_endpoint_check.setChecked(True)
        self.reverse_angle_order_check = QCheckBox("Reverse angle order")
        self.invert_angle_sign_check = QCheckBox("Invert angle sign for TIGRE")
        self.invert_angle_sign_check.setChecked(True)
        self.load_metadata_button = QPushButton("Load Metadata CSV")
        self.validate_metadata_button = QPushButton("Validate Metadata")
        layout.addWidget(QLabel("Filename column"), 0, 0)
        layout.addWidget(self.filename_column_combo, 0, 1)
        layout.addWidget(QLabel("Angle column (deg fallback)"), 1, 0)
        layout.addWidget(self.angle_column_combo, 1, 1)
        layout.addWidget(QLabel("Angle column (rad)"), 2, 0)
        layout.addWidget(self.angle_rad_column_combo, 2, 1)
        layout.addWidget(QLabel("X shift column (px)"), 3, 0)
        layout.addWidget(self.x_shift_column_combo, 3, 1)
        layout.addWidget(QLabel("Y shift column (px)"), 4, 0)
        layout.addWidget(self.y_shift_column_combo, 4, 1)
        layout.addWidget(self.remove_duplicate_endpoint_check, 5, 0, 1, 2)
        layout.addWidget(self.reverse_angle_order_check, 6, 0, 1, 2)
        layout.addWidget(self.invert_angle_sign_check, 7, 0, 1, 2)
        layout.addWidget(self.load_metadata_button, 8, 0)
        layout.addWidget(self.validate_metadata_button, 8, 1)
        parent.addWidget(group)

    def _build_preprocessing_group(self, parent: QVBoxLayout) -> None:
        group = QGroupBox("Preprocessing")
        layout = QFormLayout(group)
        self.binning_combo = QComboBox()
        self.binning_combo.addItems(["1", "2", "4"])
        self.epsilon_spin = self._double_spin(1e-12, 1.0, 1e-6, decimals=12, step=1e-6)
        self.clip_transmission_check = QCheckBox("Clip transmission")
        self.clip_transmission_check.setChecked(True)
        self.min_transmission_spin = self._double_spin(1e-12, 10.0, 1e-6, decimals=12, step=1e-6)
        self.max_transmission_spin = self._double_spin(1e-12, 100.0, 2.0, decimals=6, step=0.1)
        self.max_log_transmission_spin = self._double_spin(1e-12, 100.0, 2.0, decimals=6, step=0.1)
        self.clip_negative_attenuation_check = QCheckBox("Set negative attenuation to zero")
        self.clip_negative_attenuation_check.setChecked(True)
        self.save_transmission_check = QCheckBox("Save transmission stack")
        self.save_attenuation_check = QCheckBox("Save attenuation stack")
        self.enable_truncation_check = QCheckBox("Apply truncation correction after -ln")
        self.truncation_fraction_spin = self._double_spin(1e-6, 1.0, 0.1, decimals=6, step=0.01)
        self.compute_flat_button = QPushButton("Compute Averaged Flat Field")
        self.apply_flat_button = QPushButton("Apply Flat-Field Correction")
        self.compute_attenuation_button = QPushButton("Compute Attenuation Projections")
        layout.addRow("Binning factor", self.binning_combo)
        layout.addRow("Epsilon", self.epsilon_spin)
        layout.addRow("", self.clip_transmission_check)
        layout.addRow("Min transmission", self.min_transmission_spin)
        layout.addRow("Max transmission", self.max_transmission_spin)
        layout.addRow("Max transmission for log", self.max_log_transmission_spin)
        layout.addRow("", self.clip_negative_attenuation_check)
        layout.addRow("", self.enable_truncation_check)
        layout.addRow("Truncation extension fraction", self.truncation_fraction_spin)
        layout.addRow("", self.save_transmission_check)
        layout.addRow("", self.save_attenuation_check)
        layout.addRow(self.compute_flat_button)
        layout.addRow(self.apply_flat_button)
        layout.addRow(self.compute_attenuation_button)
        parent.addWidget(group)

    def _build_geometry_group(self, parent: QVBoxLayout) -> None:
        group = QGroupBox("TIGRE geometry")
        layout = QFormLayout(group)
        self.effective_pixel_um_spin = self._double_spin(1e-6, 1e6, 11.0, decimals=6, step=0.1)
        self.dso_mm_spin = self._double_spin(0.0, 1e6, 0.0, decimals=6, step=1.0)
        self.dsd_mm_spin = self._double_spin(0.0, 1e6, 0.0, decimals=6, step=1.0)
        self.voxel_um_spin = self._double_spin(1e-6, 1e6, 11.0, decimals=6, step=0.1)
        self.nx_spin = self._int_spin(0, 20000, 0)
        self.ny_spin = self._int_spin(0, 20000, 0)
        self.nz_spin = self._int_spin(0, 20000, 0)
        self.detector_rows_spin = self._int_spin(0, 20000, 0)
        self.detector_cols_spin = self._int_spin(0, 20000, 0)
        self.detector_rows_spin.setReadOnly(True)
        self.detector_cols_spin.setReadOnly(True)
        self.object_offset_x_spin = self._double_spin(-1e6, 1e6, 0.0, decimals=6, step=0.1)
        self.object_offset_y_spin = self._double_spin(-1e6, 1e6, 0.0, decimals=6, step=0.1)
        self.object_offset_z_spin = self._double_spin(-1e6, 1e6, 0.0, decimals=6, step=0.1)
        self.detector_offset_v_px_spin = self._double_spin(-1e6, 1e6, 0.0, decimals=6, step=0.1)
        self.detector_offset_u_px_spin = self._double_spin(-1e6, 1e6, 0.0, decimals=6, step=0.1)
        layout.addRow("Effective pixel size at object (um)", self.effective_pixel_um_spin)
        layout.addRow("Source-to-object DSO (mm)", self.dso_mm_spin)
        layout.addRow("Source-to-detector DSD (mm)", self.dsd_mm_spin)
        layout.addRow("Voxel size (um)", self.voxel_um_spin)
        layout.addRow("Nx voxels", self.nx_spin)
        layout.addRow("Ny voxels", self.ny_spin)
        layout.addRow("Nz voxels", self.nz_spin)
        layout.addRow("Detector rows", self.detector_rows_spin)
        layout.addRow("Detector columns", self.detector_cols_spin)
        layout.addRow("Object offset X (mm)", self.object_offset_x_spin)
        layout.addRow("Object offset Y (mm)", self.object_offset_y_spin)
        layout.addRow("Object offset Z (mm)", self.object_offset_z_spin)
        layout.addRow("Detector offset vertical (px)", self.detector_offset_v_px_spin)
        layout.addRow("Detector offset horizontal (px)", self.detector_offset_u_px_spin)
        parent.addWidget(group)

    def _build_orientation_group(self, parent: QVBoxLayout) -> None:
        group = QGroupBox("Orientation / center correction")
        layout = QFormLayout(group)
        self.flip_horizontal_check = QCheckBox("Flip projections horizontally")
        self.flip_vertical_check = QCheckBox("Flip projections vertically")
        self.enable_center_offset_check = QCheckBox("Enable center offset correction")
        self.center_offset_px_spin = self._double_spin(-1e6, 1e6, 0.0, decimals=6, step=0.1)
        self.center_offset_method_combo = QComboBox()
        self.center_offset_method_combo.addItem("Detector offset method", "detector_offset")
        self.center_offset_method_combo.addItem("Image shift method", "image_shift")
        note = QLabel("If center correction worsens the reconstruction, try the opposite sign.")
        note.setWordWrap(True)
        layout.addRow("", self.flip_horizontal_check)
        layout.addRow("", self.flip_vertical_check)
        layout.addRow("", self.enable_center_offset_check)
        layout.addRow("Center offset (px)", self.center_offset_px_spin)
        layout.addRow("Correction method", self.center_offset_method_combo)
        layout.addRow(note)
        parent.addWidget(group)

    def _build_center_offset_search_group(self, parent: QVBoxLayout) -> None:
        group = QGroupBox("Center Offset Search")
        layout = QFormLayout(group)
        self.center_shift_start_spin = self._double_spin(-1e6, 1e6, -10.0, decimals=4, step=0.5)
        self.center_shift_end_spin = self._double_spin(-1e6, 1e6, 10.0, decimals=4, step=0.5)
        self.center_shift_step_spin = self._double_spin(1e-6, 1e6, 1.0, decimals=4, step=0.1)
        self.center_shift_preview_slice_spin = self._int_spin(0, 20000, 0)
        self.center_shift_mode_combo = QComboBox()
        self.center_shift_mode_combo.addItem("Single central slice", "single_slice")
        self.center_shift_mode_combo.addItem("Thin band / small stack", "thin_band")
        self.center_shift_band_thickness_spin = self._int_spin(1, 101, 5)
        self.center_shift_band_thickness_spin.setEnabled(False)
        self.center_shift_sign_combo = QComboBox()
        self.center_shift_sign_combo.addItem("+1", 1.0)
        self.center_shift_sign_combo.addItem("-1", -1.0)
        self.center_shift_metric_choice_combo = QComboBox()
        self.center_shift_metric_choice_combo.addItem("Combined score", "combined_score")
        self.center_shift_metric_choice_combo.addItem("Gradient energy", "gradient_energy")
        self.center_shift_metric_choice_combo.addItem("Laplacian variance", "laplacian_variance")
        self.center_shift_metric_choice_combo.addItem("Entropy", "entropy")
        self.center_shift_per_image_contrast_check = QCheckBox("Per-image preview contrast")

        self.run_center_shift_preview_button = QPushButton("Run Manual Preview Search")
        self.run_center_shift_auto_button = QPushButton("Run Automatic Search")
        self.run_center_shift_fine_button = QPushButton("Run Fine Search Around Selected Shift")
        run_buttons = QHBoxLayout()
        run_buttons.addWidget(self.run_center_shift_preview_button)
        run_buttons.addWidget(self.run_center_shift_auto_button)

        self.center_shift_preview_slider = QSlider(Qt.Horizontal)
        self.center_shift_preview_slider.setRange(0, 0)
        self.center_shift_preview_slider.setEnabled(False)
        self.center_shift_selected_label = QLabel("No center-offset previews")
        self.center_shift_selected_label.setWordWrap(True)
        self.center_shift_auto_label = QLabel("Automatic recommendation unavailable")
        self.center_shift_auto_label.setWordWrap(True)
        self.apply_center_shift_button = QPushButton("Apply Selected Shift")
        self.apply_center_shift_button.setEnabled(False)
        self.use_auto_center_shift_button = QPushButton("Use Automatic Best Shift")
        self.use_auto_center_shift_button.setEnabled(False)
        apply_buttons = QHBoxLayout()
        apply_buttons.addWidget(self.apply_center_shift_button)
        apply_buttons.addWidget(self.use_auto_center_shift_button)

        layout.addRow("Start shift px", self.center_shift_start_spin)
        layout.addRow("End shift px", self.center_shift_end_spin)
        layout.addRow("Step px", self.center_shift_step_spin)
        layout.addRow("Preview slice index", self.center_shift_preview_slice_spin)
        layout.addRow("Preview mode", self.center_shift_mode_combo)
        layout.addRow("Band thickness", self.center_shift_band_thickness_spin)
        layout.addRow("Center-shift sign convention", self.center_shift_sign_combo)
        layout.addRow("Recommendation metric", self.center_shift_metric_choice_combo)
        layout.addRow("", self.center_shift_per_image_contrast_check)
        layout.addRow(run_buttons)
        layout.addRow(self.run_center_shift_fine_button)
        layout.addRow("Preview", self.center_shift_preview_slider)
        layout.addRow(self.center_shift_selected_label)
        layout.addRow(self.center_shift_auto_label)
        layout.addRow(apply_buttons)
        parent.addWidget(group)

    def _build_shift_group(self, parent: QVBoxLayout) -> None:
        group = QGroupBox("Per-projection shift correction")
        layout = QFormLayout(group)
        self.enable_projection_shift_check = QCheckBox("Enable per-projection shift correction")
        self.enable_projection_shift_check.setChecked(True)
        self.shift_stage_combo = QComboBox()
        self.shift_stage_combo.addItem("Before -ln: shift transmission stack", "transmission")
        self.shift_stage_combo.addItem("After -ln: shift attenuation stack", "attenuation")
        self.shift_mode_combo = QComboBox()
        for key, label, x_sign, y_sign in SHIFT_MODES:
            suffix = " [recommended first test]" if key == "B" else ""
            self.shift_mode_combo.addItem(f"{label}{suffix}", (key, x_sign, y_sign))
        self.shift_mode_combo.setCurrentIndex(0)
        self.x_shift_sign_combo = QComboBox()
        self.x_shift_sign_combo.addItem("+1: col_shift = +x_shift_px", 1)
        self.x_shift_sign_combo.addItem("-1: col_shift = -x_shift_px", -1)
        self.y_shift_sign_combo = QComboBox()
        self.y_shift_sign_combo.addItem("+1: row_shift = +y_shift_px", 1)
        self.y_shift_sign_combo.addItem("-1: row_shift = -y_shift_px", -1)
        self.y_shift_sign_combo.setCurrentIndex(0)
        self.shift_interpretation_combo = QComboBox()
        self.shift_interpretation_combo.addItem("Apply shifts as correction values", 1)
        self.shift_interpretation_combo.addItem("Apply negative of shifts as correction values", -1)
        self.shift_interpolation_order_spin = self._int_spin(0, 5, 1)
        self.shift_boundary_mode_combo = QComboBox()
        self.shift_boundary_mode_combo.addItems(["nearest", "reflect", "constant"])
        self.save_shift_previews_check = QCheckBox("Save before/after shift previews")
        self.save_shift_previews_check.setChecked(True)
        self.shift_debug_size_spin = self._int_spin(16, 4096, 256)
        self.run_shift_test_button = QPushButton("Run Shift-Convention Test")
        note = QLabel("Recommended first test for vertically inverted TIFFs: row_shift = -y_shift_px, col_shift = +x_shift_px.")
        note.setWordWrap(True)
        layout.addRow("", self.enable_projection_shift_check)
        layout.addRow("Apply shifts", self.shift_stage_combo)
        layout.addRow("Preset mode", self.shift_mode_combo)
        layout.addRow("X shift sign", self.x_shift_sign_combo)
        layout.addRow("Y shift sign", self.y_shift_sign_combo)
        layout.addRow("Shift interpretation", self.shift_interpretation_combo)
        layout.addRow("Interpolation order", self.shift_interpolation_order_spin)
        layout.addRow("Boundary mode", self.shift_boundary_mode_combo)
        layout.addRow("", self.save_shift_previews_check)
        layout.addRow("Debug volume size", self.shift_debug_size_spin)
        layout.addRow(self.run_shift_test_button)
        layout.addRow(note)
        parent.addWidget(group)

    def _build_reconstruction_group(self, parent: QVBoxLayout) -> None:
        group = QGroupBox("Reconstruction")
        layout = QFormLayout(group)
        self.run_fdk_button = QPushButton("Run FDK Reconstruction")
        self.save_reconstruction_button = QPushButton("Save Reconstruction")
        self.transpose_attenuation_check = QCheckBox("Transpose attenuation stack for TIGRE")
        self.save_reconstruction_folder_check = QCheckBox("Also save reconstruction as TIFF image stack folder")
        self.fdk_filter_combo = QComboBox()
        self.fdk_filter_combo.addItem("Ram-Lak (default)", "ram_lak")
        self.fdk_filter_combo.addItem("Shepp-Logan", "shepp_logan")
        self.fdk_filter_combo.addItem("Cosine", "cosine")
        self.fdk_filter_combo.addItem("Hamming", "hamming")
        self.fdk_filter_combo.addItem("Hann", "hann")
        self.slice_spin = self._int_spin(0, 20000, 0)
        layout.addRow(self.run_fdk_button)
        layout.addRow(self.save_reconstruction_button)
        layout.addRow("FDK filter", self.fdk_filter_combo)
        layout.addRow("", self.transpose_attenuation_check)
        layout.addRow("", self.save_reconstruction_folder_check)
        layout.addRow("Z-slice index", self.slice_spin)
        parent.addWidget(group)

    def _build_config_group(self, parent: QVBoxLayout) -> None:
        group = QGroupBox("Config")
        layout = QHBoxLayout(group)
        self.save_config_button = QPushButton("Save Config")
        self.load_config_button = QPushButton("Load Config")
        self.reset_config_button = QPushButton("Reset Parameters")
        layout.addWidget(self.save_config_button)
        layout.addWidget(self.load_config_button)
        layout.addWidget(self.reset_config_button)
        parent.addWidget(group)

    def _build_display_group(self, parent: QVBoxLayout) -> None:
        controls = QGroupBox("Image display")
        layout = QGridLayout(controls)
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(
            [
                "Raw projection",
                "Averaged flat field",
                "Transmission projection",
                "Attenuation projection",
                "Reconstructed Z slice",
                "Center offset preview",
            ]
        )
        self.projection_index_spin = self._int_spin(0, 1000000, 0)
        self.auto_contrast_check = QCheckBox("Auto contrast 1-99 percentile")
        self.auto_contrast_check.setChecked(True)
        self.manual_min_spin = self._double_spin(-1e12, 1e12, 0.0, decimals=6, step=0.1)
        self.manual_max_spin = self._double_spin(-1e12, 1e12, 1.0, decimals=6, step=0.1)
        self.image_zoom_in_button = QPushButton("Zoom In")
        self.image_zoom_out_button = QPushButton("Zoom Out")
        self.image_zoom_reset_button = QPushButton("Reset Zoom")
        self.image_zoom_label = QLabel("Zoom: 100%")
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.image_zoom_in_button)
        zoom_layout.addWidget(self.image_zoom_out_button)
        zoom_layout.addWidget(self.image_zoom_reset_button)
        zoom_layout.addWidget(self.image_zoom_label)
        layout.addWidget(QLabel("Image"), 0, 0)
        layout.addWidget(self.image_type_combo, 0, 1)
        layout.addWidget(QLabel("Projection index"), 1, 0)
        layout.addWidget(self.projection_index_spin, 1, 1)
        layout.addWidget(self.auto_contrast_check, 2, 0, 1, 2)
        layout.addWidget(QLabel("Manual min"), 3, 0)
        layout.addWidget(self.manual_min_spin, 3, 1)
        layout.addWidget(QLabel("Manual max"), 4, 0)
        layout.addWidget(self.manual_max_spin, 4, 1)
        layout.addLayout(zoom_layout, 5, 0, 1, 2)
        parent.addWidget(controls)

        self.image_figure, self.image_canvas = make_canvas(width=6.4, height=6.0)
        self.image_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        parent.addWidget(self.image_canvas, 1)
        draw_image(self.image_figure, self.image_canvas, None, "No image", zoom_factor=self.image_zoom_factor)

    def _build_right_tabs(self, tabs: QTabWidget) -> None:
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)
        self.graph_type_combo = QComboBox()
        self.graph_type_combo.addItems(
            [
                "Histogram of selected image",
                "Histogram of raw projection",
                "Histogram of averaged flat field",
                "Histogram of transmission image",
                "Histogram of attenuation image",
                "Central horizontal profile",
                "Central vertical profile",
                "Angle list plot",
                "Reconstruction slice histogram",
                "Center shift combined score",
                "Center shift gradient energy",
                "Center shift Laplacian variance",
                "Center shift entropy",
            ]
        )
        graph_layout.addWidget(self.graph_type_combo)
        self.graph_figure, self.graph_canvas = make_canvas(width=5.0, height=4.0)
        self.graph_canvas.mpl_connect("button_press_event", self._metric_plot_clicked)
        graph_layout.addWidget(self.graph_canvas, 1)
        draw_graph(self.graph_figure, self.graph_canvas, "Histogram of selected image")
        tabs.addTab(graph_tab, "Plots")

        metadata_tab = QWidget()
        metadata_layout = QVBoxLayout(metadata_tab)
        self.validation_table = QTableWidget(0, 7)
        self.validation_table.setHorizontalHeaderLabels(
            ["Index", "Filename", "Angle deg", "Angle rad", "X shift px", "Y shift px", "Exists"]
        )
        self.validation_table.horizontalHeader().setStretchLastSection(True)
        metadata_layout.addWidget(self.validation_table)
        tabs.addTab(metadata_tab, "Validation")

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        tabs.addTab(log_tab, "Log")

    def _connect_signals(self) -> None:
        self.load_metadata_button.clicked.connect(self.load_metadata)
        self.validate_metadata_button.clicked.connect(self.validate_metadata)
        self.compute_flat_button.clicked.connect(self.compute_flat_field)
        self.apply_flat_button.clicked.connect(self.apply_flat_correction)
        self.compute_attenuation_button.clicked.connect(self.compute_attenuation_stack)
        self.run_fdk_button.clicked.connect(self.run_reconstruction)
        self.run_shift_test_button.clicked.connect(self.run_shift_convention_test)
        self.save_reconstruction_button.clicked.connect(self.save_reconstruction)
        self.save_config_button.clicked.connect(self.save_config_dialog)
        self.load_config_button.clicked.connect(self.load_config_dialog)
        self.reset_config_button.clicked.connect(lambda: self.apply_config(AppConfig()))
        self.image_type_combo.currentTextChanged.connect(lambda _: self.refresh_image())
        self.projection_index_spin.valueChanged.connect(lambda _: self.refresh_image())
        self.slice_spin.valueChanged.connect(lambda _: self.refresh_image())
        self.nz_spin.valueChanged.connect(lambda _: self._sync_center_shift_preview_slice_range())
        self.auto_contrast_check.toggled.connect(lambda _: self.refresh_image())
        self.manual_min_spin.valueChanged.connect(lambda _: self.refresh_image())
        self.manual_max_spin.valueChanged.connect(lambda _: self.refresh_image())
        self.image_zoom_in_button.clicked.connect(self.zoom_image_in)
        self.image_zoom_out_button.clicked.connect(self.zoom_image_out)
        self.image_zoom_reset_button.clicked.connect(self.reset_image_zoom)
        self.graph_type_combo.currentTextChanged.connect(lambda _: self.refresh_graph())
        self.binning_combo.currentTextChanged.connect(self._clear_processed_after_binning_change)
        self.shift_mode_combo.currentIndexChanged.connect(self._apply_shift_mode_preset)
        self.center_shift_mode_combo.currentIndexChanged.connect(self._update_center_shift_mode_controls)
        self.center_shift_preview_slider.valueChanged.connect(self._center_shift_preview_changed)
        self.run_center_shift_preview_button.clicked.connect(lambda: self.run_center_shift_search(automatic=False))
        self.run_center_shift_auto_button.clicked.connect(lambda: self.run_center_shift_search(automatic=True))
        self.run_center_shift_fine_button.clicked.connect(self.run_fine_center_shift_search)
        self.apply_center_shift_button.clicked.connect(self.apply_selected_center_shift)
        self.use_auto_center_shift_button.clicked.connect(self.apply_auto_center_shift)
        self.center_shift_metric_choice_combo.currentIndexChanged.connect(self._update_auto_center_shift_recommendation)
        self.center_shift_per_image_contrast_check.toggled.connect(lambda _: self.refresh_image())
        for signal in (
            self.enable_projection_shift_check.toggled,
            self.shift_stage_combo.currentIndexChanged,
            self.x_shift_sign_combo.currentIndexChanged,
            self.y_shift_sign_combo.currentIndexChanged,
            self.shift_interpretation_combo.currentIndexChanged,
            self.shift_interpolation_order_spin.valueChanged,
            self.shift_boundary_mode_combo.currentIndexChanged,
            self.enable_truncation_check.toggled,
            self.truncation_fraction_spin.valueChanged,
            self.clip_negative_attenuation_check.toggled,
            self.max_log_transmission_spin.valueChanged,
        ):
            signal.connect(self._clear_attenuation_after_preprocessing_change)

    def _double_spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int = 4,
        step: float = 1.0,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _int_spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _browse_projection_folder(self) -> None:
        self._choose_folder(self.projection_folder_edit, "Choose projection TIFF folder")

    def _browse_reference_folder(self) -> None:
        self._choose_folder(self.reference_folder_edit, "Choose reference / flat-field TIFF folder")

    def _browse_output_folder(self) -> None:
        self._choose_folder(self.output_folder_edit, "Choose output folder")

    def _browse_metadata_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose metadata CSV", "", "CSV files (*.csv);;All files (*)")
        if path:
            self.metadata_path_edit.setText(path)

    def _browse_metadata_folder(self) -> None:
        self._choose_folder(self.metadata_path_edit, "Choose metadata folder")

    def _choose_folder(self, target: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            target.setText(path)

    def _log(self, message: str) -> None:
        line = message if message.startswith("[") else f"[{timestamp()}] {message}"
        self.log_lines.append(line)
        self.log_text.append(line)
        self.statusBar().showMessage(message[-160:])

    def _error(self, title: str, message: str) -> None:
        self._log(f"{title}: {message}")
        QMessageBox.critical(self, title, message)

    def _set_busy(self, busy: bool) -> None:
        for button in (
            self.load_metadata_button,
            self.validate_metadata_button,
            self.compute_flat_button,
            self.apply_flat_button,
            self.compute_attenuation_button,
            self.run_fdk_button,
            self.run_shift_test_button,
            self.save_reconstruction_button,
            self.save_config_button,
            self.load_config_button,
            self.reset_config_button,
            self.run_center_shift_preview_button,
            self.run_center_shift_auto_button,
            self.run_center_shift_fine_button,
            self.apply_center_shift_button,
            self.use_auto_center_shift_button,
        ):
            button.setEnabled(not busy)
        if not busy:
            self._update_center_shift_preview_controls()

    def _start_worker(
        self,
        label: str,
        fn: Callable[[Callable[[str], None]], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        if self._thread is not None:
            self._error("Busy", "A processing task is already running.")
            return
        self._log(label)
        self._set_busy(True)
        thread = QThread(self)
        worker = Worker(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._log)
        worker.finished.connect(on_success)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(self._worker_failed)
        worker.failed.connect(thread.quit)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._worker_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _worker_failed(self, text: str) -> None:
        self._log(text)
        QMessageBox.critical(self, "Processing failed", text.splitlines()[-1] if text else "Processing failed")

    def _worker_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_busy(False)
        self._save_log_if_possible()

    def _update_center_shift_mode_controls(self, *_: object) -> None:
        self.center_shift_band_thickness_spin.setEnabled(self.center_shift_mode_combo.currentData() == "thin_band")

    def _sync_center_shift_preview_slice_range(self) -> None:
        nz = self.nz_spin.value()
        maximum = max(0, nz - 1)
        current = self.center_shift_preview_slice_spin.value()
        self.center_shift_preview_slice_spin.setRange(0, 20000 if nz <= 0 else maximum)
        if nz > 0 and maximum > 0 and (current == 0 or current > maximum):
            self.center_shift_preview_slice_spin.setValue(maximum // 2)
        elif nz > 0 and current > maximum:
            self.center_shift_preview_slice_spin.setValue(maximum)

    def _update_center_shift_preview_controls(self) -> None:
        has_results = bool(self.center_shift_results)
        has_auto = self.auto_best_shift_px is not None
        self.center_shift_preview_slider.setEnabled(has_results)
        self.apply_center_shift_button.setEnabled(has_results)
        self.use_auto_center_shift_button.setEnabled(has_auto)
        self.run_center_shift_fine_button.setEnabled(has_results)
        self._update_center_shift_preview_label()

    def _update_center_shift_preview_label(self) -> None:
        result = self._selected_center_shift_result()
        if result is None:
            self.center_shift_selected_label.setText("No center-offset previews")
            return
        self.center_shift_selected_label.setText(
            f"Preview {self.current_center_shift_preview_index + 1} / {len(self.center_shift_results)}    "
            f"Shift = {result.shift_px:.4f} px    Detector offset = {result.detector_offset_physical:.6g} mm"
        )

    def _center_shift_preview_changed(self, index: int) -> None:
        if not self.center_shift_results:
            return
        self.current_center_shift_preview_index = min(max(int(index), 0), len(self.center_shift_results) - 1)
        self._update_center_shift_preview_label()
        if self.image_type_combo.currentText() != "Center offset preview":
            self.image_type_combo.setCurrentText("Center offset preview")
        else:
            self.refresh_image()
        self.refresh_graph()

    def _selected_center_shift_result(self) -> CenterShiftPreviewResult | None:
        if not self.center_shift_results:
            return None
        index = min(max(self.current_center_shift_preview_index, 0), len(self.center_shift_results) - 1)
        return self.center_shift_results[index]

    def _validate_center_shift_inputs(self) -> list[float] | None:
        try:
            shifts = generate_shift_values(
                self.center_shift_start_spin.value(),
                self.center_shift_end_spin.value(),
                self.center_shift_step_spin.value(),
            )
        except Exception as exc:
            self._error("Invalid center-offset search", str(exc))
            return None
        if len(shifts) > MAX_PREVIEW_DEFAULT_LIMIT:
            answer = QMessageBox.question(
                self,
                "Large center-offset search",
                f"This search will reconstruct {len(shifts)} previews. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return None
        elif len(shifts) > MAX_PREVIEW_SOFT_LIMIT:
            QMessageBox.warning(
                self,
                "Center-offset search",
                f"This search will reconstruct {len(shifts)} previews. A coarser step may be faster.",
            )
        return shifts

    def run_center_shift_search(self, automatic: bool = False) -> None:
        if self.attenuation_stack is None:
            self._error(
                "Attenuation required",
                "Compute attenuation projections before running center-offset preview search.",
            )
            return
        if self.validation is None:
            self._error("Metadata required", "Validate metadata before running center-offset preview search.")
            return
        try:
            params = self.geometry_params()
            detector_pixel_mm = computed_detector_pixel_mm(params)
        except Exception as exc:
            self._error("Invalid geometry", str(exc))
            return
        shifts = self._validate_center_shift_inputs()
        if not shifts:
            return

        step_value = self.center_shift_step_spin.value()
        attenuation = self.attenuation_stack
        angles_rad = self.validation.angles_rad
        preview_slice_index = self.center_shift_preview_slice_spin.value()
        preview_mode = str(self.center_shift_mode_combo.currentData() or "single_slice")
        band_thickness = self.center_shift_band_thickness_spin.value()
        center_shift_sign = float(self.center_shift_sign_combo.currentData() or 1.0)
        transpose_for_tigre = self.transpose_attenuation_check.isChecked()
        fdk_filter = self.fdk_filter()
        angle_sign = self.angle_sign_for_tigre()
        try:
            output_folder = self.output_folder()
        except Exception as exc:
            self._error("Invalid output folder", str(exc))
            return
        if not self._confirm_heavy_tigre_run(
            "Center-offset preview search",
            params,
            attenuation,
            preview_count=len(shifts),
            preview_slices=max(1, band_thickness if preview_mode == "thin_band" else 1),
        ):
            return
        mode_label = "automatic" if automatic else "manual preview"

        def job(log: Callable[[str], None]) -> dict[str, Any]:
            log("Center offset search started")
            log(f"Mode: {mode_label}")
            log(f"Start shift: {shifts[0]:.4f} px")
            log(f"End shift: {shifts[-1]:.4f} px")
            log(f"Step: {step_value:.4f} px")
            log(f"Number of previews: {len(shifts)}")
            log(f"Preview slice index: {preview_slice_index}")
            log(f"Preview mode: {preview_mode}")
            log(f"Center-shift sign convention: {center_shift_sign:+g}")
            log(f"Detector pixel size used: {detector_pixel_mm:.8g} mm/px")
            fdk_input, transpose_report = self._prepare_tigre_projection_input(
                attenuation,
                enabled=transpose_for_tigre,
                log=log,
            )
            if transpose_report:
                log("Center-offset preview uses the same TIGRE input transpose setting as full FDK.")
            results: list[CenterShiftPreviewResult] = []
            for index, shift_px in enumerate(shifts, start=1):
                detector_offset = center_shift_sign * shift_px * detector_pixel_mm
                log(
                    f"Reconstructing preview {index}/{len(shifts)}, shift = {shift_px:.4f} px, "
                    f"detector offset = {detector_offset:.8g} mm"
                )
                try:
                    preview = reconstruct_center_shift_preview(
                        fdk_input,
                        angles_rad,
                        params,
                        shift_px,
                        preview_slice_index,
                        preview_mode=preview_mode,
                        band_thickness=band_thickness,
                        center_shift_sign=center_shift_sign,
                        progress=None,
                        fdk_filter=fdk_filter,
                        angle_sign=angle_sign,
                    )
                except Exception as exc:
                    detail = str(exc)
                    if "memory" in detail.lower() or "cuda" in detail.lower():
                        detail += (
                            " Try single-slice mode, fewer shifts, a coarser step, or smaller reconstruction dimensions."
                        )
                    log(f"Center-offset preview failed for shift {shift_px:.4f} px: {detail}")
                    continue
                preview.preview_index = index - 1
                results.append(preview)
                log(f"Completed in {preview.reconstruction_time_s:.2f} s")

            if not results:
                raise RuntimeError("All center-offset previews failed. Check TIGRE/CUDA availability and geometry settings.")

            metric_table = None
            if automatic:
                metric_table = compute_metrics_for_results(results)
                for row in metric_table:
                    log(
                        f"Shift {row['shift_px']:.4f} px: "
                        f"Gradient energy = {row['gradient_energy']:.8g}, "
                        f"Laplacian variance = {row['laplacian_variance']:.8g}, "
                        f"Entropy = {row['entropy']:.8g}, "
                        f"Combined score = {row['combined_score']:.8g}"
                    )
                best_index = recommend_metric_index(metric_table, "combined_score")
                if best_index is not None:
                    log(f"Automatic recommended shift = {metric_table[best_index]['shift_px']:.4f} px")
                if output_folder is not None:
                    metric_path = output_folder / "center_offset_search_metrics.csv"
                    metric_columns = [
                        "shift_px",
                        "combined_score",
                        "gradient_energy",
                        "laplacian_variance",
                        "entropy",
                        "combined_rank",
                        "gradient_rank",
                        "laplacian_rank",
                        "entropy_rank",
                    ]
                    with metric_path.open("w", encoding="utf-8") as handle:
                        handle.write(",".join(metric_columns) + "\n")
                        for row in metric_table:
                            handle.write(",".join(f"{row[column]:.12g}" for column in metric_columns) + "\n")
                    log(f"Saved center-offset metric table: {metric_path}")
            return {"results": results, "metric_table": metric_table, "automatic": automatic}

        def success(result: dict[str, Any]) -> None:
            self.center_shift_results = result["results"]
            self.auto_metric_table = result["metric_table"]
            self.center_shift_preview_limits = self._compute_center_shift_preview_limits(self.center_shift_results)
            self.current_center_shift_preview_index = len(self.center_shift_results) // 2
            self.center_shift_preview_slider.blockSignals(True)
            self.center_shift_preview_slider.setRange(0, max(0, len(self.center_shift_results) - 1))
            self.center_shift_preview_slider.setValue(self.current_center_shift_preview_index)
            self.center_shift_preview_slider.blockSignals(False)
            if self.auto_metric_table is not None:
                self._update_auto_center_shift_recommendation()
                self.graph_type_combo.setCurrentText("Center shift combined score")
            else:
                self.auto_best_shift_px = None
                self.auto_best_preview_index = None
                self.center_shift_auto_label.setText("Automatic recommendation unavailable")
            self._update_center_shift_preview_controls()
            self.image_type_combo.setCurrentText("Center offset preview")
            self.refresh_image()

        self._start_worker(f"Running {mode_label} center-offset search.", job, success)

    def zoom_image_in(self) -> None:
        self._set_image_zoom(self.image_zoom_factor * 1.25)

    def zoom_image_out(self) -> None:
        self._set_image_zoom(self.image_zoom_factor / 1.25)

    def reset_image_zoom(self) -> None:
        self._set_image_zoom(1.0)

    def _set_image_zoom(self, zoom_factor: float) -> None:
        self.image_zoom_factor = min(max(float(zoom_factor), 1.0), 32.0)
        self._update_image_zoom_label()
        self.refresh_image()

    def _update_image_zoom_label(self) -> None:
        self.image_zoom_label.setText(f"Zoom: {self.image_zoom_factor * 100:.0f}%")
        self.image_zoom_out_button.setEnabled(self.image_zoom_factor > 1.0001)
        self.image_zoom_reset_button.setEnabled(self.image_zoom_factor > 1.0001)

    def run_fine_center_shift_search(self) -> None:
        result = self._selected_center_shift_result()
        selected_shift = result.shift_px if result is not None else self.center_offset_px_spin.value()
        previous_step = max(self.center_shift_step_spin.value(), 1e-6)
        fine_half_width = max(1.0, 2.0 * previous_step)
        fine_step = max(previous_step / 5.0, 1e-6)
        self.center_shift_start_spin.setValue(selected_shift - fine_half_width)
        self.center_shift_end_spin.setValue(selected_shift + fine_half_width)
        self.center_shift_step_spin.setValue(fine_step)
        self._log(
            "Prepared fine center-offset search: "
            f"{selected_shift - fine_half_width:.4f} to {selected_shift + fine_half_width:.4f} px, "
            f"step {fine_step:.4f} px"
        )
        self.run_center_shift_search(automatic=self.auto_metric_table is not None)

    def apply_selected_center_shift(self) -> None:
        result = self._selected_center_shift_result()
        if result is None:
            self._error("No center-offset preview", "Run a center-offset search before applying a selected shift.")
            return
        self.enable_center_offset_check.setChecked(True)
        self._set_combo_by_data(self.center_offset_method_combo, "detector_offset")
        self._set_combo_by_data(self.center_shift_sign_combo, result.center_shift_sign)
        self.center_offset_px_spin.setValue(result.shift_px)
        self._log(f"Applied selected center shift to full reconstruction settings: {result.shift_px:.4f} px")

    def apply_auto_center_shift(self) -> None:
        if self.auto_best_shift_px is None:
            self._error("No automatic recommendation", "Run automatic center-offset search before applying the best shift.")
            return
        self.enable_center_offset_check.setChecked(True)
        self._set_combo_by_data(self.center_offset_method_combo, "detector_offset")
        if self.auto_best_preview_index is not None and self.auto_best_preview_index < len(self.center_shift_results):
            self._set_combo_by_data(
                self.center_shift_sign_combo,
                self.center_shift_results[self.auto_best_preview_index].center_shift_sign,
            )
        self.center_offset_px_spin.setValue(self.auto_best_shift_px)
        self._log(f"Applied automatic center shift to full reconstruction settings: {self.auto_best_shift_px:.4f} px")

    def _update_auto_center_shift_recommendation(self, *_: object) -> None:
        if not self.auto_metric_table:
            self.auto_best_shift_px = None
            self.auto_best_preview_index = None
            self.center_shift_auto_label.setText("Automatic recommendation unavailable")
            self._update_center_shift_preview_controls()
            return
        metric_key = str(self.center_shift_metric_choice_combo.currentData() or "combined_score")
        best_index = recommend_metric_index(self.auto_metric_table, metric_key)
        if best_index is None:
            self.auto_best_shift_px = None
            self.auto_best_preview_index = None
            self.center_shift_auto_label.setText("Automatic recommendation unavailable")
            self._update_center_shift_preview_controls()
            return
        row = self.auto_metric_table[best_index]
        self.auto_best_shift_px = float(row["shift_px"])
        self.auto_best_preview_index = best_index
        self.center_shift_auto_label.setText(
            f"Automatic recommended shift: {self.auto_best_shift_px:.4f} px\n"
            f"Combined score: {row['combined_score']:.6g}    "
            f"{metric_display_name(metric_key)}: {row[metric_key]:.6g}\n"
            f"Gradient rank: {int(row['gradient_rank'])}    "
            f"Laplacian rank: {int(row['laplacian_rank'])}    "
            f"Entropy rank: {int(row['entropy_rank'])}"
        )
        self._update_center_shift_preview_controls()
        self.refresh_graph()

    def _compute_center_shift_preview_limits(
        self,
        results: list[CenterShiftPreviewResult],
    ) -> tuple[float, float] | None:
        finite_chunks = [result.preview_image[np.isfinite(result.preview_image)].ravel() for result in results]
        finite_chunks = [chunk for chunk in finite_chunks if chunk.size]
        if not finite_chunks:
            return None
        pixels = np.concatenate(finite_chunks)
        low, high = np.percentile(pixels, [1.0, 99.0])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.nanmin(pixels))
            high = float(np.nanmax(pixels))
        if high <= low:
            high = low + 1.0
        return float(low), float(high)

    def _confirm_heavy_tigre_run(
        self,
        title: str,
        params: GeometryParams,
        projection_stack: np.ndarray,
        preview_count: int = 1,
        preview_slices: int | None = None,
    ) -> bool:
        projection_gb = projection_stack.nbytes / (1024**3)
        if preview_slices is None:
            voxel_count = int(params.Nx) * int(params.Ny) * int(params.Nz)
            output_gb = voxel_count * 4 / (1024**3)
            workload_label = f"Output volume: {params.Nx} x {params.Ny} x {params.Nz} ({output_gb:.2f} GB float32)"
            heavy = output_gb >= 2.0 or projection_gb >= 2.0 or voxel_count >= 512**3
        else:
            preview_voxels = int(params.Nx) * int(params.Ny) * int(preview_slices)
            output_gb = preview_voxels * 4 / (1024**3)
            workload_label = (
                f"Preview slab: {params.Nx} x {params.Ny} x {preview_slices} per shift "
                f"({output_gb:.3f} GB float32 output each)"
            )
            heavy = (
                projection_gb >= 1.0
                and preview_count >= 5
                and int(params.Nx) * int(params.Ny) >= 768 * 768
            ) or preview_count > MAX_PREVIEW_SOFT_LIMIT

        if not heavy:
            return True

        message = (
            f"{title} may take a long time in TIGRE's native/CUDA code.\n\n"
            f"Projection stack: {tuple(projection_stack.shape)} ({projection_gb:.2f} GB float32)\n"
            f"{workload_label}\n"
            f"Number of TIGRE FDK calls: {preview_count}\n\n"
            "While TIGRE is running, avoid Ctrl+C, the IDE stop button, or closing the terminal. "
            "The Intel Fortran runtime may abort the whole process with forrtl error (200).\n\n"
            "Continue?"
        )
        answer = QMessageBox.question(
            self,
            "Large TIGRE job",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            self._log(f"Cancelled {title} before TIGRE launch.")
            return False
        self._log(
            f"Confirmed large TIGRE job: {title}; projection stack {tuple(projection_stack.shape)}, "
            f"{projection_gb:.2f} GB float32, FDK calls={preview_count}."
        )
        return True

    def load_metadata(self) -> None:
        try:
            path = find_metadata_csv(self.metadata_path_edit.text().strip())
            frame = load_metadata_csv(path)
        except Exception as exc:
            self._error("Metadata load failed", str(exc))
            return
        self.metadata_frame = frame
        self.metadata_path_edit.setText(str(path))
        self.filename_column_combo.clear()
        self.angle_column_combo.clear()
        self.angle_rad_column_combo.clear()
        self.x_shift_column_combo.clear()
        self.y_shift_column_combo.clear()
        columns = [str(column) for column in frame.columns]
        self.filename_column_combo.addItems(columns)
        self.angle_column_combo.addItems(columns)
        self.angle_rad_column_combo.addItem("No radian column / use degrees", "")
        self.x_shift_column_combo.addItem("No X shift column", "")
        self.y_shift_column_combo.addItem("No Y shift column", "")
        for column in columns:
            self.angle_rad_column_combo.addItem(column, column)
            self.x_shift_column_combo.addItem(column, column)
            self.y_shift_column_combo.addItem(column, column)
        filename_guess = suggest_column(columns, FILENAME_HINTS)
        angle_guess = suggest_column(columns, ANGLE_HINTS)
        angle_rad_guess = self._suggest_optional_column(columns, ANGLE_RAD_HINTS)
        x_shift_guess = self._suggest_optional_column(columns, X_SHIFT_HINTS)
        y_shift_guess = self._suggest_optional_column(columns, Y_SHIFT_HINTS)
        self.filename_column_combo.setCurrentText(filename_guess)
        self.angle_column_combo.setCurrentText(angle_guess)
        if angle_rad_guess:
            angle_rad_index = self.angle_rad_column_combo.findData(angle_rad_guess)
            if angle_rad_index >= 0:
                self.angle_rad_column_combo.setCurrentIndex(angle_rad_index)
        if x_shift_guess:
            x_shift_index = self.x_shift_column_combo.findData(x_shift_guess)
            if x_shift_index >= 0:
                self.x_shift_column_combo.setCurrentIndex(x_shift_index)
        if y_shift_guess:
            y_shift_index = self.y_shift_column_combo.findData(y_shift_guess)
            if y_shift_index >= 0:
                self.y_shift_column_combo.setCurrentIndex(y_shift_index)
        self._log(f"Loaded metadata CSV: {path}")
        self._log(f"Metadata columns: {', '.join(columns)}")

    def validate_metadata(self) -> None:
        if self.metadata_frame is None:
            self.load_metadata()
            if self.metadata_frame is None:
                return
        try:
            validation = validate_metadata(
                self.metadata_frame,
                self.projection_folder_edit.text().strip(),
                self.filename_column_combo.currentText(),
                self.angle_column_combo.currentText(),
                self.angle_rad_column(),
                self.x_shift_column(),
                self.y_shift_column(),
                remove_duplicate_endpoint=self.remove_duplicate_endpoint_check.isChecked(),
                reverse_angle_order=self.reverse_angle_order_check.isChecked(),
            )
            self._populate_validation_table(validation)
            if validation.duplicate_filenames:
                raise ValueError(f"Duplicate projection filenames in metadata: {validation.duplicate_filenames[:10]}")
            if validation.missing_files:
                raise FileNotFoundError(f"Missing projection files: {validation.missing_files[:10]}")
            first = read_tiff_float32(validation.records[0].path, binning=1)
            binned_first = read_tiff_float32(validation.records[0].path, binning=self.binning_factor())
        except Exception as exc:
            self.validation = None
            self._error("Metadata validation failed", str(exc))
            return

        self.validation = validation
        self.expected_raw_shape = first.shape
        rows, cols = binned_first.shape
        self.detector_rows_spin.setValue(rows)
        self.detector_cols_spin.setValue(cols)
        if self.nx_spin.value() == 0:
            self.nx_spin.setValue(cols)
        if self.ny_spin.value() == 0:
            self.ny_spin.setValue(cols)
        if self.nz_spin.value() == 0:
            self.nz_spin.setValue(rows)
        self._sync_center_shift_preview_slice_range()
        self.projection_index_spin.setRange(0, max(0, len(validation.records) - 1))
        self._log(f"Validated {len(validation.records)} projection/angle pairs.")
        self._log(f"Projection image size after binning: rows={rows}, cols={cols}")
        memory_gb = estimate_stack_memory_gb(len(validation.records), rows, cols)
        self._log(f"Estimated float32 projection stack memory: {memory_gb:.3f} GB")
        if memory_gb > 8.0:
            QMessageBox.warning(
                self,
                "Large dataset",
                f"The projection stack alone is estimated at {memory_gb:.2f} GB as float32. "
                "Consider 2x or 4x binning for a test run.",
            )
        self._log(f"Metadata CSV path: {self.metadata_path_edit.text().strip()}")
        self._log(f"Filename column: {self.filename_column_combo.currentText()}")
        self._log(f"Angle degree fallback column: {self.angle_column_combo.currentText()}")
        self._log(f"Angle radian column: {self.angle_rad_column() or '(none selected)'}")
        self._log(f"FDK angle input source: {validation.angle_input_source}")
        self._log(f"X shift column: {self.x_shift_column() or '(none selected)'}, found={validation.x_shift_column_found}")
        self._log(f"Y shift column: {self.y_shift_column() or '(none selected)'}, found={validation.y_shift_column_found}")
        self._log(f"Angle range in degrees: {validation.angle_min_deg:.8g} to {validation.angle_max_deg:.8g}")
        self._log(f"Angle range in radians: {validation.angle_min_rad:.8g} to {validation.angle_max_rad:.8g}")
        self._log(f"Angle direction: {validation.angle_direction}")
        self._log(f"Invert angle sign for TIGRE: {self.invert_angle_sign_check.isChecked()}")
        self._log(f"Duplicate endpoint angle detected: {validation.duplicate_endpoint_detected}")
        for warning in validation.warnings:
            self._log(f"Warning: {warning}")
        self.image_type_combo.setCurrentText("Raw projection")
        self.refresh_image()

    def _populate_validation_table(self, validation: MetadataValidation) -> None:
        rows = validation.table_rows
        self.validation_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, key in enumerate(
                ("index", "filename", "angle_deg", "angle_rad", "x_shift_px", "y_shift_px", "exists")
            ):
                self.validation_table.setItem(row_index, column_index, QTableWidgetItem(row[key]))
        self.validation_table.resizeColumnsToContents()

    def compute_flat_field(self) -> None:
        if self.expected_raw_shape is None:
            self.validate_metadata()
            if self.expected_raw_shape is None:
                return
        reference_folder = self.reference_folder_edit.text().strip()
        binning = self.binning_factor()
        expected_shape = self.expected_raw_shape
        output_folder = self.output_folder()

        def job(log: Callable[[str], None]) -> dict[str, Any]:
            flat, count = compute_average_flat_field(reference_folder, expected_shape, binning, log)
            saved_path = None
            if output_folder is not None:
                saved_path = save_image(output_folder / "averaged_flat_field.tif", flat)
                log(f"Saved averaged flat field: {saved_path}")
            return {"flat": flat, "count": count, "saved_path": saved_path}

        def success(result: dict[str, Any]) -> None:
            self.flat_field = result["flat"]
            self._log(f"Number of reference images averaged: {result['count']}")
            self.image_type_combo.setCurrentText("Averaged flat field")
            self.refresh_image()

        self._start_worker("Computing averaged flat field.", job, success)

    def apply_flat_correction(self) -> None:
        if self.validation is None:
            self.validate_metadata()
            if self.validation is None:
                return
        if self.flat_field is None:
            self._error("Flat field required", "Compute the averaged flat field before applying correction.")
            return
        records = self.validation.records
        binning = self.binning_factor()
        flat = self.flat_field
        output_folder = self.output_folder()
        settings = {
            "epsilon": self.epsilon_spin.value(),
            "clip_transmission": self.clip_transmission_check.isChecked(),
            "min_transmission": self.min_transmission_spin.value(),
            "max_transmission": self.max_transmission_spin.value(),
            "flip_horizontal": self.flip_horizontal_check.isChecked(),
            "flip_vertical": self.flip_vertical_check.isChecked(),
            "save": self.save_transmission_check.isChecked(),
        }

        def job(log: Callable[[str], None]) -> dict[str, Any]:
            raw = load_projection_stack(records, binning=binning, progress=log)
            log(f"Number of projections loaded: {raw.shape[0]}")
            log(f"Projection stack shape: {raw.shape}")
            log(f"Flip horizontal: {settings['flip_horizontal']}")
            log(f"Flip vertical: {settings['flip_vertical']}")
            transmission = apply_flat_field(
                raw,
                flat,
                epsilon=settings["epsilon"],
                clip_transmission=settings["clip_transmission"],
                min_transmission=settings["min_transmission"],
                max_transmission=settings["max_transmission"],
                flip_horizontal=settings["flip_horizontal"],
                flip_vertical=settings["flip_vertical"],
            )
            log("Computed transmission stack with T = I/F.")
            if settings["clip_transmission"]:
                log(
                    f"Transmission clipping enabled: "
                    f"{settings['min_transmission']:g} to {settings['max_transmission']:g}"
                )
            else:
                log("Transmission clipping disabled.")
            saved_path = None
            if settings["save"] and output_folder is not None:
                saved_path = save_stack_folder(output_folder / "transmission_stack", transmission, "transmission", log)
            return {"raw": raw, "transmission": transmission, "saved_path": saved_path}

        def success(result: dict[str, Any]) -> None:
            self.raw_stack = result["raw"]
            self.transmission_stack = result["transmission"]
            self.attenuation_stack = None
            self.attenuation_preprocessing_report = []
            self.reconstruction = None
            self.image_type_combo.setCurrentText("Transmission projection")
            self.refresh_image()

        self._start_worker("Applying flat-field correction.", job, success)

    def compute_attenuation_stack(self) -> None:
        if self.transmission_stack is None:
            self._error("Transmission required", "Apply flat-field correction before computing attenuation.")
            return
        if self.enable_projection_shift_check.isChecked():
            if self.validation is None:
                self.validate_metadata()
                if self.validation is None:
                    return
            if self.validation.x_shift_px is None or self.validation.y_shift_px is None:
                self._error(
                    "Shift columns required",
                    "Select and validate complete x_shift_px and y_shift_px columns before enabling shift correction.",
                )
                return
        transmission = self.transmission_stack
        validation = self.validation
        output_folder = self.output_folder()
        x_sign, y_sign, interpretation_sign = self.shift_signs_from_ui()
        shift_enabled = self.enable_projection_shift_check.isChecked()
        shift_stage = self.shift_stage()
        try:
            params_for_report = self.geometry_params()
        except Exception:
            params_for_report = GeometryParams(
                effective_pixel_size_um=self.effective_pixel_um_spin.value(),
                DSO_mm=max(self.dso_mm_spin.value(), 1.0),
                DSD_mm=max(self.dsd_mm_spin.value(), 2.0),
                Nx=max(self.nx_spin.value(), 1),
                Ny=max(self.ny_spin.value(), 1),
                Nz=max(self.nz_spin.value(), 1),
                voxel_size_um=self.voxel_um_spin.value(),
                enable_center_offset=self.enable_center_offset_check.isChecked(),
                center_offset_px=self.center_offset_px_spin.value(),
                center_offset_method=self.center_offset_method_combo.currentData(),
                center_shift_sign=float(self.center_shift_sign_combo.currentData() or 1.0),
            )
        settings = {
            "epsilon": self.epsilon_spin.value(),
            "max_for_log": self.max_log_transmission_spin.value(),
            "clip_negative": self.clip_negative_attenuation_check.isChecked(),
            "save": self.save_attenuation_check.isChecked(),
            "shift_enabled": shift_enabled,
            "shift_stage": shift_stage,
            "x_sign": x_sign,
            "y_sign": y_sign,
            "interpretation_sign": interpretation_sign,
            "interpolation_order": self.shift_interpolation_order_spin.value(),
            "boundary_mode": self.shift_boundary_mode_combo.currentText(),
            "save_previews": self.save_shift_previews_check.isChecked(),
            "metadata_csv": self.metadata_path_edit.text().strip(),
            "flip_horizontal": self.flip_horizontal_check.isChecked(),
            "flip_vertical": self.flip_vertical_check.isChecked(),
            "truncation_enabled": self.enable_truncation_check.isChecked(),
            "truncation_fraction": self.truncation_fraction_spin.value(),
        }

        def job(log: Callable[[str], None]) -> dict[str, Any]:
            preprocessing_report = self._build_preprocessing_report(validation, params_for_report, settings)
            working_transmission = transmission
            if settings["shift_enabled"] and settings["shift_stage"] == "transmission":
                assert validation is not None
                log("Applying per-projection shifts to transmission stack before -ln.")
                effective_x_sign, effective_y_sign = effective_shift_signs(
                    settings["x_sign"],
                    settings["y_sign"],
                    settings["interpretation_sign"],
                )
                working_transmission = apply_projection_shifts(
                    transmission,
                    validation.x_shift_px,
                    validation.y_shift_px,
                    x_sign=effective_x_sign,
                    y_sign=effective_y_sign,
                    interpolation_order=settings["interpolation_order"],
                    mode=settings["boundary_mode"],
                    logger=log,
                )
                if output_folder is not None and settings["save_previews"]:
                    save_shift_debug_previews(
                        output_folder,
                        validation.records,
                        transmission,
                        working_transmission,
                        logger=log,
                        stage_label="transmission",
                    )
            elif settings["shift_enabled"]:
                log("Per-projection shifts will be applied after -ln to the attenuation stack.")
            else:
                log("Per-projection shift correction disabled for attenuation computation.")

            attenuation = compute_attenuation(
                working_transmission,
                epsilon=settings["epsilon"],
                max_transmission_for_log=settings["max_for_log"],
                clip_negative_to_zero=settings["clip_negative"],
            )
            log("Computed attenuation stack with p = -ln(T).")
            log(f"Set negative attenuation to zero: {settings['clip_negative']}")
            if settings["shift_enabled"] and settings["shift_stage"] == "attenuation":
                assert validation is not None
                log("Applying per-projection shifts to attenuation stack after -ln.")
                effective_x_sign, effective_y_sign = effective_shift_signs(
                    settings["x_sign"],
                    settings["y_sign"],
                    settings["interpretation_sign"],
                )
                before_shift = attenuation
                attenuation = apply_projection_shifts(
                    attenuation,
                    validation.x_shift_px,
                    validation.y_shift_px,
                    x_sign=effective_x_sign,
                    y_sign=effective_y_sign,
                    interpolation_order=settings["interpolation_order"],
                    mode=settings["boundary_mode"],
                    logger=log,
                )
                if output_folder is not None and settings["save_previews"]:
                    save_shift_debug_previews(
                        output_folder,
                        validation.records,
                        before_shift,
                        attenuation,
                        logger=log,
                        stage_label="attenuation",
                    )
            if settings["truncation_enabled"]:
                before_shape = attenuation.shape
                attenuation = truncation_correction(attenuation, settings["truncation_fraction"])
                log(
                    "Applied truncation correction after -ln: "
                    f"extension_fraction={settings['truncation_fraction']:g}, shape {before_shape} -> {attenuation.shape}"
                )
                preprocessing_report.extend(
                    [
                        "",
                        "Truncation correction",
                        "---------------------",
                        "Truncation correction enabled: True",
                        f"Extension fraction: {settings['truncation_fraction']:g}",
                        "Method: symmetric left/right detector-column padding with ramp taper",
                        f"Shape before truncation correction: {before_shape}",
                        f"Shape after truncation correction: {attenuation.shape}",
                    ]
                )
            else:
                log("Truncation correction disabled.")
                preprocessing_report.extend(
                    [
                        "",
                        "Truncation correction",
                        "---------------------",
                        "Truncation correction enabled: False",
                    ]
                )
            saved_path = None
            if settings["save"] and output_folder is not None:
                saved_path = save_stack_folder(output_folder / "attenuation_stack", attenuation, "attenuation", log)
            return {"attenuation": attenuation, "saved_path": saved_path, "preprocessing_report": preprocessing_report}

        def success(result: dict[str, Any]) -> None:
            self.attenuation_stack = result["attenuation"]
            self.attenuation_preprocessing_report = result["preprocessing_report"]
            self.reconstruction = None
            self._clear_center_shift_results()
            self.image_type_combo.setCurrentText("Attenuation projection")
            self.refresh_image()

        self._start_worker("Computing attenuation projections.", job, success)

    def run_reconstruction(self) -> None:
        if self.attenuation_stack is None:
            self._error("Attenuation required", "Compute attenuation projections before running FDK.")
            return
        if self.validation is None:
            self._error("Metadata required", "Validate metadata before running FDK.")
            return
        try:
            params = self.geometry_params()
        except Exception as exc:
            self._error("Invalid geometry", str(exc))
            return
        attenuation = self.attenuation_stack
        if not self._confirm_heavy_tigre_run("Full FDK reconstruction", params, attenuation):
            return
        angles_rad = self.validation.angles_rad
        validation = self.validation
        output_folder = self.output_folder()
        fdk_filter = self.fdk_filter()
        angle_sign = self.angle_sign_for_tigre()
        save_reconstruction_stack_folder = self.save_reconstruction_folder_check.isChecked()
        x_sign, y_sign, interpretation_sign = self.shift_signs_from_ui()
        shift_settings = {
            "enabled": self.enable_projection_shift_check.isChecked(),
            "shift_enabled": self.enable_projection_shift_check.isChecked(),
            "shift_stage": self.shift_stage(),
            "mode_key": self.shift_mode_key(),
            "x_sign": x_sign,
            "y_sign": y_sign,
            "interpretation_sign": interpretation_sign,
            "interpolation_order": self.shift_interpolation_order_spin.value(),
            "boundary_mode": self.shift_boundary_mode_combo.currentText(),
            "save_previews": self.save_shift_previews_check.isChecked(),
            "flip_horizontal": self.flip_horizontal_check.isChecked(),
            "flip_vertical": self.flip_vertical_check.isChecked(),
            "metadata_csv": self.metadata_path_edit.text().strip(),
            "transpose_for_tigre": self.transpose_attenuation_check.isChecked(),
        }

        def job(log: Callable[[str], None]) -> dict[str, Any]:
            log(f"Angular range in degrees: {validation.angle_min_deg:.8g} to {validation.angle_max_deg:.8g}")
            log(f"Angular range in radians: {validation.angle_min_rad:.8g} to {validation.angle_max_rad:.8g}")
            log(f"FDK angle input source: {validation.angle_input_source}")
            log(f"TIGRE angle sign multiplier: {angle_sign:+d}")
            log(f"FDK reconstruction start time: {timestamp()}")
            preprocessing_report = list(self.attenuation_preprocessing_report)
            if not preprocessing_report:
                preprocessing_report = self._build_preprocessing_report(validation, params, shift_settings)
                preprocessing_report.append(
                    "Warning: attenuation preprocessing report was rebuilt from current GUI settings; "
                    "recompute attenuation after changing preprocessing options."
                )
            fdk_input = attenuation
            log("Using already computed attenuation stack for FDK. Recompute attenuation after changing shift/truncation settings.")
            fdk_input, transpose_report = self._prepare_tigre_projection_input(
                fdk_input,
                enabled=shift_settings["transpose_for_tigre"],
                log=log,
            )
            result = run_fdk(
                fdk_input,
                angles_rad,
                params,
                progress=log,
                extra_report_lines=transpose_report + preprocessing_report,
                fdk_filter=fdk_filter,
                angle_sign=angle_sign,
            )
            log(f"FDK reconstruction end time: {result.finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
            report_path = None
            saved_path = None
            saved_folder = None
            if output_folder is not None:
                report_path = write_processing_log(output_folder / "geometry_sanity_report.txt", result.geometry_report)
                log(f"Saved geometry sanity report: {report_path}")
                saved_path = save_stack_tiff(output_folder / "reconstruction.tif", result.volume)
                log(f"Output reconstruction TIFF stack path: {saved_path}")
                if save_reconstruction_stack_folder:
                    saved_folder = save_stack_folder(
                        output_folder / "reconstruction_stack",
                        result.volume,
                        "reconstruction",
                        log,
                    )
                    log(f"Output reconstruction TIFF image stack folder: {saved_folder}")
            return {
                "volume": result.volume,
                "saved_path": saved_path,
                "saved_folder": saved_folder,
                "report_path": report_path,
            }

        def success(result: dict[str, Any]) -> None:
            self.reconstruction = result["volume"]
            self.slice_spin.setRange(0, max(0, self.reconstruction.shape[0] - 1))
            self.slice_spin.setValue(min(self.slice_spin.value(), self.reconstruction.shape[0] - 1))
            self.image_type_combo.setCurrentText("Reconstructed Z slice")
            self.refresh_image()

        self._start_worker("Running TIGRE FDK reconstruction.", job, success)

    def run_shift_convention_test(self) -> None:
        if self.transmission_stack is None:
            self._error("Transmission required", "Apply flat-field correction before running shift-convention tests.")
            return
        if self.validation is None:
            self._error("Metadata required", "Validate metadata before running shift-convention tests.")
            return
        if self.validation.x_shift_px is None or self.validation.y_shift_px is None:
            self._error("Shift columns required", "Select and validate complete x_shift_px and y_shift_px columns first.")
            return
        try:
            params = self.geometry_params()
        except Exception as exc:
            self._error("Invalid geometry", str(exc))
            return

        transmission = self.transmission_stack
        validation = self.validation
        angles_rad = validation.angles_rad
        output_root = self.output_folder() or Path.cwd()
        angle_sign = self.angle_sign_for_tigre()
        debug_size = self.shift_debug_size_spin.value()
        interpolation_order = self.shift_interpolation_order_spin.value()
        boundary_mode = self.shift_boundary_mode_combo.currentText()
        shift_stage = self.shift_stage()
        flip_horizontal = self.flip_horizontal_check.isChecked()
        flip_vertical = self.flip_vertical_check.isChecked()
        transpose_for_tigre = self.transpose_attenuation_check.isChecked()
        metadata_csv = self.metadata_path_edit.text().strip()
        fdk_filter = self.fdk_filter()
        debug_params = replace(params, Nx=debug_size, Ny=debug_size, Nz=debug_size)
        base_config = self.config_from_ui().to_dict()
        attenuation_settings = {
            "epsilon": self.epsilon_spin.value(),
            "max_for_log": self.max_log_transmission_spin.value(),
            "clip_negative": self.clip_negative_attenuation_check.isChecked(),
            "truncation_enabled": self.enable_truncation_check.isChecked(),
            "truncation_fraction": self.truncation_fraction_spin.value(),
        }

        def job(log: Callable[[str], None]) -> dict[str, Any]:
            folder = output_root / "shift_mode_tests"
            folder.mkdir(parents=True, exist_ok=True)
            log(f"Running four-mode shift-convention test with debug volume {debug_size} x {debug_size} x {debug_size}.")
            log(f"Shift-convention test stage: {shift_stage}")
            log(f"TIGRE angle sign multiplier for shift-convention test: {angle_sign:+d}")
            log("Shift-convention test uses correction-value formulas: no extra negative interpretation multiplier.")
            saved: list[str] = []
            center_offset_mm = self._center_offset_mm(debug_params)
            for key, _label, x_sign, y_sign in SHIFT_MODES:
                stem = self._shift_mode_file_stem(key, x_sign, y_sign)
                log(f"Testing shift mode {key}: row_sign={y_sign:+d}, col_sign={x_sign:+d}")
                if shift_stage == "transmission":
                    shifted_transmission = apply_projection_shifts(
                        transmission,
                        validation.x_shift_px,
                        validation.y_shift_px,
                        x_sign=x_sign,
                        y_sign=y_sign,
                        interpolation_order=interpolation_order,
                        mode=boundary_mode,
                        logger=log,
                    )
                    shifted = compute_attenuation(
                        shifted_transmission,
                        epsilon=attenuation_settings["epsilon"],
                        max_transmission_for_log=attenuation_settings["max_for_log"],
                        clip_negative_to_zero=attenuation_settings["clip_negative"],
                    )
                else:
                    base_attenuation = compute_attenuation(
                        transmission,
                        epsilon=attenuation_settings["epsilon"],
                        max_transmission_for_log=attenuation_settings["max_for_log"],
                        clip_negative_to_zero=attenuation_settings["clip_negative"],
                    )
                    shifted = apply_projection_shifts(
                        base_attenuation,
                        validation.x_shift_px,
                        validation.y_shift_px,
                        x_sign=x_sign,
                        y_sign=y_sign,
                        interpolation_order=interpolation_order,
                        mode=boundary_mode,
                        logger=log,
                    )
                truncation_report = [
                    "",
                    "Truncation correction",
                    "---------------------",
                    f"Truncation correction enabled: {attenuation_settings['truncation_enabled']}",
                ]
                if attenuation_settings["truncation_enabled"]:
                    before_shape = shifted.shape
                    shifted = truncation_correction(shifted, attenuation_settings["truncation_fraction"])
                    log(
                        f"Applied truncation correction for mode {key}: "
                        f"extension_fraction={attenuation_settings['truncation_fraction']:g}, "
                        f"shape {before_shape} -> {shifted.shape}"
                    )
                    truncation_report.extend(
                        [
                            f"Extension fraction: {attenuation_settings['truncation_fraction']:g}",
                            "Method: symmetric left/right detector-column padding with ramp taper",
                            f"Shape before truncation correction: {before_shape}",
                            f"Shape after truncation correction: {shifted.shape}",
                        ]
                    )
                fdk_input, transpose_report = self._prepare_tigre_projection_input(
                    shifted,
                    enabled=transpose_for_tigre,
                    log=log,
                )
                shift_report = build_shift_sanity_report(
                    shift_source_csv=metadata_csv,
                    x_column=validation.x_shift_column,
                    y_column=validation.y_shift_column,
                    x_column_found=validation.x_shift_column_found,
                    y_column_found=validation.y_shift_column_found,
                    x_shift_px=validation.x_shift_px,
                    y_shift_px=validation.y_shift_px,
                    enabled=True,
                    shift_stage=shift_stage,
                    x_sign=x_sign,
                    y_sign=y_sign,
                    interpretation_sign=1,
                    interpolation_order=interpolation_order,
                    boundary_mode=boundary_mode,
                    flip_vertical=flip_vertical,
                    flip_horizontal=flip_horizontal,
                    center_offset_enabled=debug_params.enable_center_offset,
                    center_offset_px=debug_params.center_offset_px,
                    center_offset_mm=center_offset_mm,
                    center_offset_method=debug_params.center_offset_method,
                    duplicate_endpoint_removed=validation.duplicate_endpoint_removed,
                )
                result = run_fdk(
                    fdk_input,
                    angles_rad,
                    debug_params,
                    progress=log,
                    extra_report_lines=transpose_report + shift_report + truncation_report,
                    fdk_filter=fdk_filter,
                    angle_sign=angle_sign,
                )
                center_slice = result.volume[result.volume.shape[0] // 2]
                slice_path = save_png_image(folder / f"{stem}_slice.png", center_slice)
                config_path = self._save_shift_mode_config(
                    folder / f"{stem}_config.json",
                    base_config,
                    key,
                    x_sign,
                    y_sign,
                )
                report_path = write_processing_log(folder / f"{stem}_geometry_sanity_report.txt", result.geometry_report)
                log(f"Saved shift mode {key} central slice: {slice_path}")
                saved.extend([str(slice_path), str(config_path), str(report_path)])
            return {"folder": folder, "saved": saved}

        def success(result: dict[str, Any]) -> None:
            self._log(f"Shift-convention test finished. Results folder: {result['folder']}")

        self._start_worker("Running shift-convention test.", job, success)

    def save_reconstruction(self) -> None:
        if self.reconstruction is None:
            self._error("No reconstruction", "Run FDK before saving the reconstruction.")
            return
        default = self.output_folder() or Path.cwd()
        answer = QMessageBox.question(
            self,
            "Choose reconstruction save format",
            "Save the reconstruction as a folder of individual TIFF slices?\n\n"
            "Yes: one TIFF image per reconstructed Z slice.\n"
            "No: one multi-page TIFF volume.",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No,
        )
        if answer == QMessageBox.Cancel:
            return
        try:
            if answer == QMessageBox.Yes:
                folder_path = QFileDialog.getExistingDirectory(
                    self,
                    "Choose reconstruction TIFF slice folder",
                    str(default / "reconstruction_stack"),
                )
                if not folder_path:
                    return
                folder = Path(folder_path)
                saved = save_stack_folder(folder, self.reconstruction, "reconstruction", self._log)
                self._log(f"Saved reconstruction TIFF image stack folder: {saved}")
            else:
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save reconstruction TIFF stack",
                    str(default / "reconstruction.tif"),
                    "Multi-page TIFF stack (*.tif *.tiff);;All files (*)",
                )
                if not path:
                    return
                saved = save_stack_tiff(path, self.reconstruction)
                self._log(f"Saved reconstruction TIFF stack: {saved}")
        except Exception as exc:
            self._error("Save failed", str(exc))
            return
        self._save_log_if_possible()

    def refresh_image(self) -> None:
        image, title = self.current_image()
        auto_contrast = self.auto_contrast_check.isChecked()
        manual_min = self.manual_min_spin.value()
        manual_max = self.manual_max_spin.value()
        if (
            self.image_type_combo.currentText() == "Center offset preview"
            and self.center_shift_results
            and not self.center_shift_per_image_contrast_check.isChecked()
            and self.center_shift_preview_limits is not None
        ):
            auto_contrast = False
            manual_min, manual_max = self.center_shift_preview_limits
        draw_image(
            self.image_figure,
            self.image_canvas,
            image,
            title,
            auto_contrast=auto_contrast,
            manual_min=manual_min,
            manual_max=manual_max,
            zoom_factor=self.image_zoom_factor,
        )
        self.refresh_graph()

    def current_image(self) -> tuple[np.ndarray | None, str]:
        kind = self.image_type_combo.currentText()
        proj_index = self.projection_index_spin.value()
        if kind == "Raw projection":
            image = self._raw_projection_image(proj_index)
            return image, f"Raw projection {proj_index}"
        if kind == "Averaged flat field":
            return self.flat_field, "Averaged flat field"
        if kind == "Transmission projection":
            image = self._stack_image(self.transmission_stack, proj_index)
            return image, f"Transmission projection {proj_index}"
        if kind == "Attenuation projection":
            image = self._stack_image(self.attenuation_stack, proj_index)
            return image, f"Attenuation projection {proj_index}"
        if kind == "Reconstructed Z slice":
            if self.reconstruction is None:
                return None, "Reconstructed Z slice"
            z = min(self.slice_spin.value(), self.reconstruction.shape[0] - 1)
            return self.reconstruction[z], f"Reconstructed Z slice {z}"
        if kind == "Center offset preview":
            result = self._selected_center_shift_result()
            if result is None:
                return None, "Center offset preview"
            return (
                result.preview_image,
                f"Center offset preview: shift = {result.shift_px:.4f} px | "
                f"Preview {self.current_center_shift_preview_index + 1}/{len(self.center_shift_results)}",
            )
        return None, "No image"

    def _raw_projection_image(self, index: int) -> np.ndarray | None:
        if self.raw_stack is not None:
            return self._stack_image(self.raw_stack, index)
        if self.validation is None:
            return None
        index = min(max(index, 0), len(self.validation.records) - 1)
        try:
            return read_tiff_float32(self.validation.records[index].path, binning=self.binning_factor())
        except Exception as exc:
            self._log(f"Could not read raw projection for display: {exc}")
            return None

    def _stack_image(self, stack: np.ndarray | None, index: int) -> np.ndarray | None:
        if stack is None:
            return None
        index = min(max(index, 0), stack.shape[0] - 1)
        return stack[index]

    def refresh_graph(self) -> None:
        graph_type = self.graph_type_combo.currentText()
        metric_key = self._center_shift_metric_key_for_graph(graph_type)
        if metric_key is not None:
            self._draw_center_shift_metric_graph(metric_key)
            return
        image = self.graph_image(graph_type)
        angles = None
        if self.validation is not None:
            angles = np.asarray([record.angle_deg for record in self.validation.records], dtype=np.float32)
        draw_graph(self.graph_figure, self.graph_canvas, graph_type, image=image, angles_deg=angles)

    def graph_image(self, graph_type: str) -> np.ndarray | None:
        proj_index = self.projection_index_spin.value()
        if graph_type == "Angle list plot":
            return None
        if graph_type == "Histogram of raw projection":
            return self._raw_projection_image(proj_index)
        if graph_type == "Histogram of averaged flat field":
            return self.flat_field
        if graph_type == "Histogram of transmission image":
            return self._stack_image(self.transmission_stack, proj_index)
        if graph_type == "Histogram of attenuation image":
            return self._stack_image(self.attenuation_stack, proj_index)
        if graph_type == "Reconstruction slice histogram":
            if self.reconstruction is None:
                return None
            return self.reconstruction[min(self.slice_spin.value(), self.reconstruction.shape[0] - 1)]
        image, _ = self.current_image()
        return image

    def _center_shift_metric_key_for_graph(self, graph_type: str) -> str | None:
        return {
            "Center shift combined score": "combined_score",
            "Center shift gradient energy": "gradient_energy",
            "Center shift Laplacian variance": "laplacian_variance",
            "Center shift entropy": "entropy",
        }.get(graph_type)

    def _draw_center_shift_metric_graph(self, metric_key: str) -> None:
        if not self.auto_metric_table:
            draw_center_shift_metric_plot(
                self.graph_figure,
                self.graph_canvas,
                np.asarray([], dtype=np.float32),
                np.asarray([], dtype=np.float32),
                metric_display_name(metric_key),
            )
            return
        shifts = np.asarray([row["shift_px"] for row in self.auto_metric_table], dtype=np.float64)
        values = np.asarray([row.get(metric_key, np.nan) for row in self.auto_metric_table], dtype=np.float64)
        selected = self._selected_center_shift_result()
        selected_shift = selected.shift_px if selected is not None else None
        draw_center_shift_metric_plot(
            self.graph_figure,
            self.graph_canvas,
            shifts,
            values,
            metric_display_name(metric_key),
            selected_shift=selected_shift,
            recommended_shift=self.auto_best_shift_px,
        )

    def _metric_plot_clicked(self, event: object) -> None:
        graph_type = self.graph_type_combo.currentText()
        if self._center_shift_metric_key_for_graph(graph_type) is None:
            return
        xdata = getattr(event, "xdata", None)
        if xdata is None or not self.center_shift_results:
            return
        shifts = np.asarray([result.shift_px for result in self.center_shift_results], dtype=np.float64)
        if shifts.size == 0:
            return
        closest = int(np.argmin(np.abs(shifts - float(xdata))))
        self.center_shift_preview_slider.setValue(closest)

    def geometry_params(self) -> GeometryParams:
        return GeometryParams(
            effective_pixel_size_um=self.effective_pixel_um_spin.value(),
            DSO_mm=self.dso_mm_spin.value(),
            DSD_mm=self.dsd_mm_spin.value(),
            Nx=self.nx_spin.value(),
            Ny=self.ny_spin.value(),
            Nz=self.nz_spin.value(),
            voxel_size_um=self.voxel_um_spin.value(),
            object_offset_x_mm=self.object_offset_x_spin.value(),
            object_offset_y_mm=self.object_offset_y_spin.value(),
            object_offset_z_mm=self.object_offset_z_spin.value(),
            detector_offset_vertical_px=self.detector_offset_v_px_spin.value(),
            detector_offset_horizontal_px=self.detector_offset_u_px_spin.value(),
            enable_center_offset=self.enable_center_offset_check.isChecked(),
            center_offset_px=self.center_offset_px_spin.value(),
            center_offset_method=self.center_offset_method_combo.currentData(),
            center_shift_sign=float(self.center_shift_sign_combo.currentData() or 1.0),
        )

    def _center_offset_mm(self, params: GeometryParams) -> float:
        if params.DSO_mm <= 0 or params.DSD_mm <= 0:
            return 0.0
        return params.center_shift_sign * params.center_offset_px * computed_detector_pixel_mm(params)

    def _prepare_tigre_projection_input(
        self,
        stack: np.ndarray,
        enabled: bool,
        log: Callable[[str], None] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        before_shape = tuple(stack.shape)
        lines = [
            "",
            "TIGRE projection axis mapping",
            "-----------------------------",
            f"Transpose attenuation stack for TIGRE: {enabled}",
            f"Input shape before optional transpose: {before_shape}",
        ]
        if not enabled:
            lines.append("FDK receives attenuation stack as [n_angles, rows, cols].")
            if log is not None:
                log("TIGRE attenuation transpose disabled; using [n_angles, rows, cols].")
            return stack, lines

        transposed = np.ascontiguousarray(np.transpose(stack, (0, 2, 1)), dtype=np.float32)
        after_shape = tuple(transposed.shape)
        lines.extend(
            [
                "Applied attenuation transpose: np.transpose(atten, (0, 2, 1))",
                f"FDK input shape after transpose: {after_shape}",
                "Axis meaning changed from [n_angles, rows, cols] to [n_angles, cols, rows].",
                "For square detectors geo.nDetector may stay numerically the same, but row/column meaning is swapped.",
            ]
        )
        if self.enable_center_offset_check.isChecked() and self.center_offset_method_combo.currentData() == "image_shift":
            lines.append("Warning: image-shift center correction runs after transpose and shifts axis 2 of the transposed stack.")
        if log is not None:
            log(f"Transposed attenuation stack for TIGRE: {before_shape} -> {after_shape}")
        return transposed, lines

    def binning_factor(self) -> int:
        return int(self.binning_combo.currentText())

    def angle_rad_column(self) -> str:
        data = self.angle_rad_column_combo.currentData()
        return str(data) if data else ""

    def x_shift_column(self) -> str:
        data = self.x_shift_column_combo.currentData()
        return str(data) if data else ""

    def y_shift_column(self) -> str:
        data = self.y_shift_column_combo.currentData()
        return str(data) if data else ""

    def shift_mode_key(self) -> str:
        data = self.shift_mode_combo.currentData()
        return str(data[0]) if data else "B"

    def shift_stage(self) -> str:
        data = self.shift_stage_combo.currentData()
        return str(data) if data else "transmission"

    def fdk_filter(self) -> str:
        data = self.fdk_filter_combo.currentData()
        return str(data) if data else "ram_lak"

    def angle_sign_for_tigre(self) -> int:
        return -1 if self.invert_angle_sign_check.isChecked() else 1

    def shift_signs_from_ui(self) -> tuple[int, int, int]:
        x_sign = int(self.x_shift_sign_combo.currentData())
        y_sign = int(self.y_shift_sign_combo.currentData())
        interpretation_sign = int(self.shift_interpretation_combo.currentData())
        return x_sign, y_sign, interpretation_sign

    def _apply_shift_mode_preset(self, *_: object) -> None:
        data = self.shift_mode_combo.currentData()
        if not data:
            return
        _, x_sign, y_sign = data
        self._set_combo_by_data(self.x_shift_sign_combo, int(x_sign))
        self._set_combo_by_data(self.y_shift_sign_combo, int(y_sign))

    def _set_combo_by_data(self, combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _build_preprocessing_report(
        self,
        validation: MetadataValidation | None,
        params: GeometryParams,
        settings: dict[str, Any],
    ) -> list[str]:
        lines = [
            "",
            "Zeiss preprocessing order",
            "-------------------------",
            "Current order: transmission = I/F",
            f"Per-projection shift stage: {settings['shift_stage']}",
            "Then attenuation = -ln(transmission)",
            "Then optional truncation correction",
            "xrmreader reference order: divide_by_reference -> revert_shifts -> negative_logarithm -> truncation_correction",
        ]
        if validation is None:
            lines.extend(
                build_shift_sanity_report(
                    shift_source_csv=settings["metadata_csv"],
                    x_column="",
                    y_column="",
                    x_column_found=False,
                    y_column_found=False,
                    x_shift_px=None,
                    y_shift_px=None,
                    enabled=settings["shift_enabled"],
                    shift_stage=settings["shift_stage"],
                    x_sign=settings["x_sign"],
                    y_sign=settings["y_sign"],
                    interpretation_sign=settings["interpretation_sign"],
                    interpolation_order=settings["interpolation_order"],
                    boundary_mode=settings["boundary_mode"],
                    flip_vertical=settings["flip_vertical"],
                    flip_horizontal=settings["flip_horizontal"],
                    center_offset_enabled=params.enable_center_offset,
                    center_offset_px=params.center_offset_px,
                    center_offset_mm=self._center_offset_mm(params),
                    center_offset_method=params.center_offset_method,
                    duplicate_endpoint_removed=False,
                )
            )
            return lines
        lines.extend(
            build_shift_sanity_report(
                shift_source_csv=settings["metadata_csv"],
                x_column=validation.x_shift_column,
                y_column=validation.y_shift_column,
                x_column_found=validation.x_shift_column_found,
                y_column_found=validation.y_shift_column_found,
                x_shift_px=validation.x_shift_px,
                y_shift_px=validation.y_shift_px,
                enabled=settings["shift_enabled"],
                shift_stage=settings["shift_stage"],
                x_sign=settings["x_sign"],
                y_sign=settings["y_sign"],
                interpretation_sign=settings["interpretation_sign"],
                interpolation_order=settings["interpolation_order"],
                boundary_mode=settings["boundary_mode"],
                flip_vertical=settings["flip_vertical"],
                flip_horizontal=settings["flip_horizontal"],
                center_offset_enabled=params.enable_center_offset,
                center_offset_px=params.center_offset_px,
                center_offset_mm=self._center_offset_mm(params),
                center_offset_method=params.center_offset_method,
                duplicate_endpoint_removed=validation.duplicate_endpoint_removed,
            )
        )
        return lines

    def _set_shift_mode(self, key: str) -> None:
        for index in range(self.shift_mode_combo.count()):
            data = self.shift_mode_combo.itemData(index)
            if data and data[0] == key:
                self.shift_mode_combo.setCurrentIndex(index)
                return

    def _shift_mode_file_stem(self, key: str, x_sign: int, y_sign: int) -> str:
        row = "plus_y" if y_sign > 0 else "minus_y"
        col = "plus_x" if x_sign > 0 else "minus_x"
        return f"mode_{key}_row_{row}_col_{col}"

    def _save_shift_mode_config(
        self,
        path: str | Path,
        base_config: dict[str, Any],
        key: str,
        x_sign: int,
        y_sign: int,
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        values = dict(base_config)
        values.update(
            {
                "shift_test_mode": key,
                "shift_test_x_sign": x_sign,
                "shift_test_y_sign": y_sign,
                "shift_test_formula": (
                    f"row_shift = {'+' if y_sign > 0 else '-'}y_shift_px, "
                    f"col_shift = {'+' if x_sign > 0 else '-'}x_shift_px"
                ),
            }
        )
        with output.open("w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        return output

    def _suggest_optional_column(self, columns: list[str], hints: tuple[str, ...]) -> str:
        normalized = [(column, column.lower().strip()) for column in columns]
        for hint in hints:
            for original, lower in normalized:
                if lower == hint:
                    return original
        for hint in hints:
            for original, lower in normalized:
                if hint in lower:
                    return original
        return ""

    def output_folder(self) -> Path | None:
        text = self.output_folder_edit.text().strip()
        if not text:
            return None
        path = Path(text)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _clear_processed_after_binning_change(self) -> None:
        self.flat_field = None
        self.raw_stack = None
        self.transmission_stack = None
        self.attenuation_stack = None
        self.attenuation_preprocessing_report = []
        self.reconstruction = None
        self._clear_center_shift_results()
        self.validation = None
        self.expected_raw_shape = None
        self.detector_rows_spin.setValue(0)
        self.detector_cols_spin.setValue(0)
        self._log("Binning changed; cleared loaded/processed arrays. Validate metadata again.")
        self.refresh_image()

    def _clear_attenuation_after_preprocessing_change(self, *_: object) -> None:
        if self.attenuation_stack is None and self.reconstruction is None and not self.attenuation_preprocessing_report:
            return
        self.attenuation_stack = None
        self.attenuation_preprocessing_report = []
        self.reconstruction = None
        self._clear_center_shift_results()
        self._log("Preprocessing settings changed; recompute attenuation before running FDK.")
        self.refresh_image()

    def _clear_center_shift_results(self) -> None:
        self.center_shift_results = []
        self.current_center_shift_preview_index = 0
        self.auto_best_shift_px = None
        self.auto_best_preview_index = None
        self.auto_metric_table = None
        self.center_shift_preview_limits = None
        self.center_shift_preview_slider.blockSignals(True)
        self.center_shift_preview_slider.setRange(0, 0)
        self.center_shift_preview_slider.setValue(0)
        self.center_shift_preview_slider.blockSignals(False)
        self.center_shift_auto_label.setText("Automatic recommendation unavailable")
        self._update_center_shift_preview_controls()

    def config_from_ui(self) -> AppConfig:
        return AppConfig(
            projection_folder=self.projection_folder_edit.text().strip(),
            reference_folder=self.reference_folder_edit.text().strip(),
            metadata_csv=self.metadata_path_edit.text().strip(),
            output_folder=self.output_folder_edit.text().strip(),
            filename_column=self.filename_column_combo.currentText(),
            angle_column=self.angle_column_combo.currentText(),
            angle_rad_column=self.angle_rad_column(),
            x_shift_column=self.x_shift_column(),
            y_shift_column=self.y_shift_column(),
            effective_pixel_size_um=self.effective_pixel_um_spin.value(),
            DSO_mm=self.dso_mm_spin.value() or None,
            DSD_mm=self.dsd_mm_spin.value() or None,
            Nx=self.nx_spin.value() or None,
            Ny=self.ny_spin.value() or None,
            Nz=self.nz_spin.value() or None,
            voxel_size_um=self.voxel_um_spin.value(),
            detector_rows=self.detector_rows_spin.value() or None,
            detector_columns=self.detector_cols_spin.value() or None,
            object_offset_x_mm=self.object_offset_x_spin.value(),
            object_offset_y_mm=self.object_offset_y_spin.value(),
            object_offset_z_mm=self.object_offset_z_spin.value(),
            detector_offset_vertical_px=self.detector_offset_v_px_spin.value(),
            detector_offset_horizontal_px=self.detector_offset_u_px_spin.value(),
            flip_horizontal=self.flip_horizontal_check.isChecked(),
            flip_vertical=self.flip_vertical_check.isChecked(),
            enable_center_offset=self.enable_center_offset_check.isChecked(),
            center_offset_px=self.center_offset_px_spin.value(),
            center_offset_method=self.center_offset_method_combo.currentData(),
            center_shift_sign=float(self.center_shift_sign_combo.currentData() or 1.0),
            epsilon=self.epsilon_spin.value(),
            clip_transmission=self.clip_transmission_check.isChecked(),
            min_transmission=self.min_transmission_spin.value(),
            max_transmission=self.max_transmission_spin.value(),
            max_transmission_for_log=self.max_log_transmission_spin.value(),
            clip_negative_attenuation_to_zero=self.clip_negative_attenuation_check.isChecked(),
            remove_duplicate_endpoint_angle=self.remove_duplicate_endpoint_check.isChecked(),
            reverse_angle_order=self.reverse_angle_order_check.isChecked(),
            invert_angle_sign_for_tigre=self.invert_angle_sign_check.isChecked(),
            binning_factor=self.binning_factor(),
            save_transmission_stack=self.save_transmission_check.isChecked(),
            save_attenuation_stack=self.save_attenuation_check.isChecked(),
            enable_projection_shift_correction=self.enable_projection_shift_check.isChecked(),
            shift_mode_preset=self.shift_mode_key(),
            x_shift_sign=int(self.x_shift_sign_combo.currentData()),
            y_shift_sign=int(self.y_shift_sign_combo.currentData()),
            shift_interpretation_sign=int(self.shift_interpretation_combo.currentData()),
            shift_interpolation_order=self.shift_interpolation_order_spin.value(),
            shift_boundary_mode=self.shift_boundary_mode_combo.currentText(),
            save_shift_debug_previews=self.save_shift_previews_check.isChecked(),
            shift_debug_volume_size=self.shift_debug_size_spin.value(),
            projection_shift_stage=self.shift_stage(),
            enable_truncation_correction=self.enable_truncation_check.isChecked(),
            truncation_extension_fraction=self.truncation_fraction_spin.value(),
            transpose_attenuation_for_tigre=self.transpose_attenuation_check.isChecked(),
            fdk_filter=self.fdk_filter(),
            save_reconstruction_stack_folder=self.save_reconstruction_folder_check.isChecked(),
        )

    def apply_config(self, config: AppConfig) -> None:
        self.projection_folder_edit.setText(config.projection_folder)
        self.reference_folder_edit.setText(config.reference_folder)
        self.metadata_path_edit.setText(config.metadata_csv)
        self.output_folder_edit.setText(config.output_folder)
        self.effective_pixel_um_spin.setValue(config.effective_pixel_size_um)
        self.dso_mm_spin.setValue(config.DSO_mm or 0.0)
        self.dsd_mm_spin.setValue(config.DSD_mm or 0.0)
        self.voxel_um_spin.setValue(config.voxel_size_um)
        self.nx_spin.setValue(config.Nx or 0)
        self.ny_spin.setValue(config.Ny or 0)
        self.nz_spin.setValue(config.Nz or 0)
        self._sync_center_shift_preview_slice_range()
        self.detector_rows_spin.setValue(config.detector_rows or 0)
        self.detector_cols_spin.setValue(config.detector_columns or 0)
        self.object_offset_x_spin.setValue(config.object_offset_x_mm)
        self.object_offset_y_spin.setValue(config.object_offset_y_mm)
        self.object_offset_z_spin.setValue(config.object_offset_z_mm)
        self.detector_offset_v_px_spin.setValue(config.detector_offset_vertical_px)
        self.detector_offset_u_px_spin.setValue(config.detector_offset_horizontal_px)
        self.flip_horizontal_check.setChecked(config.flip_horizontal)
        self.flip_vertical_check.setChecked(config.flip_vertical)
        self.enable_center_offset_check.setChecked(config.enable_center_offset)
        self.center_offset_px_spin.setValue(config.center_offset_px)
        self.center_offset_method_combo.setCurrentIndex(
            max(0, self.center_offset_method_combo.findData(config.center_offset_method))
        )
        self._set_combo_by_data(self.center_shift_sign_combo, config.center_shift_sign)
        self.epsilon_spin.setValue(config.epsilon)
        self.clip_transmission_check.setChecked(config.clip_transmission)
        self.min_transmission_spin.setValue(config.min_transmission)
        self.max_transmission_spin.setValue(config.max_transmission)
        self.max_log_transmission_spin.setValue(config.max_transmission_for_log)
        self.clip_negative_attenuation_check.setChecked(config.clip_negative_attenuation_to_zero)
        self.remove_duplicate_endpoint_check.setChecked(config.remove_duplicate_endpoint_angle)
        self.reverse_angle_order_check.setChecked(config.reverse_angle_order)
        self.invert_angle_sign_check.setChecked(config.invert_angle_sign_for_tigre)
        self.binning_combo.setCurrentText(str(config.binning_factor))
        self.save_transmission_check.setChecked(config.save_transmission_stack)
        self.save_attenuation_check.setChecked(config.save_attenuation_stack)
        self.enable_projection_shift_check.setChecked(config.enable_projection_shift_correction)
        self._set_shift_mode(config.shift_mode_preset)
        self._set_combo_by_data(self.x_shift_sign_combo, config.x_shift_sign)
        self._set_combo_by_data(self.y_shift_sign_combo, config.y_shift_sign)
        self._set_combo_by_data(self.shift_interpretation_combo, config.shift_interpretation_sign)
        self.shift_interpolation_order_spin.setValue(config.shift_interpolation_order)
        self.shift_boundary_mode_combo.setCurrentText(config.shift_boundary_mode)
        self.save_shift_previews_check.setChecked(config.save_shift_debug_previews)
        self.shift_debug_size_spin.setValue(config.shift_debug_volume_size)
        self._set_combo_by_data(self.shift_stage_combo, config.projection_shift_stage)
        self.enable_truncation_check.setChecked(config.enable_truncation_correction)
        self.truncation_fraction_spin.setValue(config.truncation_extension_fraction)
        self.transpose_attenuation_check.setChecked(config.transpose_attenuation_for_tigre)
        self._set_combo_by_data(self.fdk_filter_combo, config.fdk_filter)
        self.save_reconstruction_folder_check.setChecked(config.save_reconstruction_stack_folder)
        self._log("Loaded parameters into the GUI.")
        if config.metadata_csv and Path(config.metadata_csv).exists():
            self.load_metadata()
            if config.filename_column:
                self.filename_column_combo.setCurrentText(config.filename_column)
            if config.angle_column:
                self.angle_column_combo.setCurrentText(config.angle_column)
            if config.angle_rad_column:
                angle_rad_index = self.angle_rad_column_combo.findData(config.angle_rad_column)
                if angle_rad_index >= 0:
                    self.angle_rad_column_combo.setCurrentIndex(angle_rad_index)
            if config.x_shift_column:
                x_shift_index = self.x_shift_column_combo.findData(config.x_shift_column)
                if x_shift_index >= 0:
                    self.x_shift_column_combo.setCurrentIndex(x_shift_index)
            if config.y_shift_column:
                y_shift_index = self.y_shift_column_combo.findData(config.y_shift_column)
                if y_shift_index >= 0:
                    self.y_shift_column_combo.setCurrentIndex(y_shift_index)

    def save_config_dialog(self) -> None:
        default_folder = self.output_folder() or Path.cwd()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save config JSON",
            str(default_folder / "config.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            saved = save_config(self.config_from_ui(), path)
        except Exception as exc:
            self._error("Config save failed", str(exc))
            return
        self._log(f"Saved config: {saved}")

    def load_config_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load config JSON", "", "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            config = load_config(path)
        except Exception as exc:
            self._error("Config load failed", str(exc))
            return
        self.apply_config(config)
        self._log(f"Loaded config: {path}")

    def _maybe_offer_recent_config(self) -> None:
        path = recent_config_path()
        if path is None:
            return
        answer = QMessageBox.question(
            self,
            "Load recent config?",
            f"Load the recent reconstruction config?\n\n{path}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            try:
                self.apply_config(load_config(path))
                self._log(f"Loaded recent config: {path}")
            except Exception as exc:
                self._error("Recent config failed", str(exc))

    def _save_log_if_possible(self) -> None:
        folder = self.output_folder()
        if folder is None:
            return
        try:
            write_processing_log(folder / "processing_log.txt", self.log_lines)
        except Exception as exc:
            self._log(f"Could not save processing log: {exc}")


def run_app(argv: list[str] | None = None) -> int:
    app = QApplication(argv or [])
    app.setStyleSheet(APP_STYLE_SHEET)
    window = MicroCTReconstructionWindow()
    window.show()
    return exec_app(app)
