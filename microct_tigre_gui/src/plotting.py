from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


APP_BACKGROUND = "#273A59"
PANEL_BACKGROUND = "#1F2F49"
TEXT_COLOR = "#F4F8FF"
MUTED_TEXT_COLOR = "#DDE8F7"
GRID_COLOR = "#8EAEE0"
PRIMARY_PLOT_COLOR = "#78A6E6"
SECONDARY_PLOT_COLOR = "#FF8A80"
ACCENT_PLOT_COLOR = "#80D6B6"


def make_canvas(width: float = 5.0, height: float = 4.0, dpi: int = 100) -> tuple[Figure, FigureCanvas]:
    figure = Figure(figsize=(width, height), dpi=dpi, facecolor=APP_BACKGROUND)
    canvas = FigureCanvas(figure)
    canvas.setStyleSheet(f"background-color: {APP_BACKGROUND};")
    return figure, canvas


def style_axis(axis) -> None:
    axis.set_facecolor(PANEL_BACKGROUND)
    axis.title.set_color(TEXT_COLOR)
    axis.xaxis.label.set_color(MUTED_TEXT_COLOR)
    axis.yaxis.label.set_color(MUTED_TEXT_COLOR)
    axis.tick_params(colors=MUTED_TEXT_COLOR)
    for spine in axis.spines.values():
        spine.set_color(GRID_COLOR)


def contrast_limits(image: np.ndarray, auto: bool, manual_min: float, manual_max: float) -> tuple[float, float]:
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    if auto:
        low, high = np.percentile(finite, [1, 99])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(finite))
            high = float(np.max(finite))
    else:
        low, high = float(manual_min), float(manual_max)
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def draw_image(
    figure: Figure,
    canvas: FigureCanvas,
    image: np.ndarray | None,
    title: str,
    auto_contrast: bool = True,
    manual_min: float = 0.0,
    manual_max: float = 1.0,
    zoom_factor: float = 1.0,
) -> None:
    figure.clear()
    figure.set_facecolor(APP_BACKGROUND)
    axis = figure.add_subplot(111)
    style_axis(axis)
    if image is None:
        axis.text(0.5, 0.5, "No image", ha="center", va="center", color=TEXT_COLOR, transform=axis.transAxes)
        axis.set_axis_off()
    else:
        low, high = contrast_limits(image, auto_contrast, manual_min, manual_max)
        view = axis.imshow(image, cmap="gray", vmin=low, vmax=high, origin="upper")
        axis.set_title(title, color=TEXT_COLOR)
        apply_image_zoom(axis, image, zoom_factor)
        axis.set_axis_off()
        colorbar = figure.colorbar(view, ax=axis, fraction=0.046, pad=0.04)
        colorbar.ax.set_facecolor(APP_BACKGROUND)
        colorbar.ax.tick_params(colors=MUTED_TEXT_COLOR)
        colorbar.outline.set_edgecolor(GRID_COLOR)
    figure.tight_layout()
    canvas.draw_idle()


def apply_image_zoom(axis, image: np.ndarray, zoom_factor: float) -> None:
    zoom = max(1.0, float(zoom_factor))
    if zoom <= 1.0:
        return
    rows, cols = np.asarray(image).shape[:2]
    if rows <= 0 or cols <= 0:
        return
    center_x = (cols - 1) / 2.0
    center_y = (rows - 1) / 2.0
    half_width = max(0.5, cols / (2.0 * zoom))
    half_height = max(0.5, rows / (2.0 * zoom))
    axis.set_xlim(center_x - half_width, center_x + half_width)
    axis.set_ylim(center_y + half_height, center_y - half_height)


def draw_graph(
    figure: Figure,
    canvas: FigureCanvas,
    graph_type: str,
    image: np.ndarray | None = None,
    angles_deg: np.ndarray | None = None,
) -> None:
    figure.clear()
    figure.set_facecolor(APP_BACKGROUND)
    axis = figure.add_subplot(111)
    style_axis(axis)
    if graph_type == "Angle list plot":
        if angles_deg is None or len(angles_deg) == 0:
            axis.text(0.5, 0.5, "No angles", ha="center", va="center", color=TEXT_COLOR, transform=axis.transAxes)
        else:
            axis.plot(np.arange(len(angles_deg)), angles_deg, linewidth=1.2, color=PRIMARY_PLOT_COLOR)
            axis.scatter(np.arange(len(angles_deg)), angles_deg, s=10, color=ACCENT_PLOT_COLOR)
            axis.set_xlabel("Projection index")
            axis.set_ylabel("Angle (deg)")
            axis.grid(True, color=GRID_COLOR, alpha=0.25)
    elif image is None:
        axis.text(0.5, 0.5, "No image", ha="center", va="center", color=TEXT_COLOR, transform=axis.transAxes)
    elif "Histogram" in graph_type:
        finite = image[np.isfinite(image)]
        if finite.size:
            axis.hist(finite.ravel(), bins=128, color=PRIMARY_PLOT_COLOR, alpha=0.9)
            axis.set_xlabel("Intensity")
            axis.set_ylabel("Count")
        else:
            axis.text(
                0.5,
                0.5,
                "No finite pixels",
                ha="center",
                va="center",
                color=TEXT_COLOR,
                transform=axis.transAxes,
            )
    elif graph_type == "Central horizontal profile":
        row = image.shape[0] // 2
        axis.plot(image[row, :], color=PRIMARY_PLOT_COLOR)
        axis.set_xlabel("u column")
        axis.set_ylabel("Value")
        axis.grid(True, color=GRID_COLOR, alpha=0.25)
    elif graph_type == "Central vertical profile":
        col = image.shape[1] // 2
        axis.plot(image[:, col], color=SECONDARY_PLOT_COLOR)
        axis.set_xlabel("v row")
        axis.set_ylabel("Value")
        axis.grid(True, color=GRID_COLOR, alpha=0.25)
    else:
        finite = image[np.isfinite(image)]
        if finite.size:
            axis.hist(finite.ravel(), bins=128, color=ACCENT_PLOT_COLOR, alpha=0.9)
            axis.set_xlabel("Value")
            axis.set_ylabel("Count")
    axis.set_title(graph_type, color=TEXT_COLOR)
    style_axis(axis)
    figure.tight_layout()
    canvas.draw_idle()


def draw_center_shift_metric_plot(
    figure: Figure,
    canvas: FigureCanvas,
    shifts: np.ndarray,
    values: np.ndarray,
    metric_label: str,
    selected_shift: float | None = None,
    recommended_shift: float | None = None,
) -> None:
    figure.clear()
    figure.set_facecolor(APP_BACKGROUND)
    axis = figure.add_subplot(111)
    style_axis(axis)
    finite = np.isfinite(shifts) & np.isfinite(values)
    if not np.any(finite):
        axis.text(0.5, 0.5, "No metric values", ha="center", va="center", color=TEXT_COLOR, transform=axis.transAxes)
    else:
        x = shifts[finite]
        y = values[finite]
        axis.plot(x, y, color=PRIMARY_PLOT_COLOR, linewidth=1.4)
        axis.scatter(x, y, color=ACCENT_PLOT_COLOR, s=24, zorder=3)
        if selected_shift is not None and np.isfinite(selected_shift):
            axis.axvline(selected_shift, color=SECONDARY_PLOT_COLOR, linewidth=1.4, linestyle="--", label="Selected")
        if recommended_shift is not None and np.isfinite(recommended_shift):
            best_index = int(np.argmin(np.abs(x - recommended_shift)))
            axis.scatter(
                [x[best_index]],
                [y[best_index]],
                color="#FFFFFF",
                edgecolor=SECONDARY_PLOT_COLOR,
                linewidth=1.3,
                s=64,
                zorder=4,
                label="Recommended",
            )
        axis.set_xlabel("Center shift (px)")
        axis.set_ylabel(metric_label)
        axis.grid(True, color=GRID_COLOR, alpha=0.25)
        legend = axis.legend(loc="best")
        if legend is not None:
            legend.get_frame().set_facecolor(PANEL_BACKGROUND)
            legend.get_frame().set_edgecolor(GRID_COLOR)
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)
    axis.set_title(f"{metric_label} vs center shift", color=TEXT_COLOR)
    style_axis(axis)
    figure.tight_layout()
    canvas.draw_idle()
