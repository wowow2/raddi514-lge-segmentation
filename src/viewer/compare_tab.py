"""Compare tab: two runs on the same case and slice, side by side."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.config import LABEL_NAMES, LABEL_SCAR, repo_path

from . import qt_theme
from .model import ViewerModel, contour_color, error_rgba, label_rgba, montage_cols
from .theme import DARK
from .widgets import ClassToggles, Field, SegmentedControl


class CompareTab(QWidget):
    """Linked A/B panes for method-vs-method inspection."""

    status = pyqtSignal(str)

    def __init__(self, model: ViewerModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.model = model
        self.colors = {i: DARK.classes[i - 1][1] for i in range(1, 5)}
        self.visible = {1, 2, 3, 4}
        self.focus_label = LABEL_SCAR
        self.overlay = "Prediction"
        self.style = "Mask"
        self.opacity = 0.70
        self.montage = False
        self._axis_slices: dict[object, int] = {}
        self._slice_guard = False
        self._build()
        self.refresh(reset_slice=True)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(9)

        runs_row = QHBoxLayout()
        runs_row.setSpacing(14)
        self.run_a = QComboBox()
        self.run_b = QComboBox()
        for combo in (self.run_a, self.run_b):
            combo.addItem("GT only", None)
            for choice in self.model.run_choices(only_preds=True):
                combo.addItem(choice.name, choice.run_id)
        pred_runs = self.model.pred_run_ids()
        if pred_runs:
            default = self.model.default_pred_run()
            self._select(self.run_a, default)
            other = next((r for r in pred_runs if r != default), None)
            self._select(self.run_b, other)
        self.run_a.currentIndexChanged.connect(lambda _i: self._render())
        self.run_b.currentIndexChanged.connect(lambda _i: self._render())
        runs_row.addWidget(Field("Left", self.run_a, stretch=True))
        runs_row.addWidget(Field("Right", self.run_b, stretch=True))
        runs_row.addStretch(0)
        root.addLayout(runs_row)

        controls = QHBoxLayout()
        controls.setSpacing(14)
        self.overlay_combo = QComboBox()
        for name in ("Prediction", "GT", "Both", "Error"):
            self.overlay_combo.addItem(name, name)
        self.overlay_combo.currentTextChanged.connect(self._on_overlay)
        controls.addWidget(Field("Overlay", self.overlay_combo))

        self.style_control = SegmentedControl(["Mask", "Contour", "Both"], self.style)
        self.style_control.changed.connect(self._on_style)
        controls.addWidget(Field("Style", self.style_control))

        self.focus_combo = QComboBox()
        for label in (1, 2, 3, 4):
            self.focus_combo.addItem(LABEL_NAMES[label], label)
        self.focus_combo.setCurrentIndex(2)
        self.focus_combo.currentIndexChanged.connect(self._on_focus)
        controls.addWidget(Field("Focus", self.focus_combo))

        self.class_toggles = ClassToggles()
        self.class_toggles.changed.connect(self._on_classes)
        controls.addWidget(self.class_toggles)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(int(self.opacity * 100))
        self.opacity_slider.setFixedWidth(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        controls.addWidget(Field("Opacity", self.opacity_slider))

        controls.addStretch(1)
        self.layout_control = SegmentedControl(["Single", "Montage"], "Single")
        self.layout_control.changed.connect(self._on_layout)
        controls.addWidget(Field("Layout", self.layout_control))
        root.addLayout(controls)

        slice_row = QHBoxLayout()
        slice_row.setSpacing(10)
        label = QLabel("Slice")
        label.setObjectName("Muted")
        slice_row.addWidget(label)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self._on_slice)
        slice_row.addWidget(self.slice_slider, 1)
        self.slice_text = QLabel("1 / 1")
        self.slice_text.setObjectName("Muted")
        self.slice_text.setMinimumWidth(56)
        slice_row.addWidget(self.slice_text)
        root.addLayout(slice_row)

        self.figure = Figure(figsize=(9.5, 7.2), dpi=100, facecolor=DARK.figure_face)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setStyleSheet("background: transparent;")
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        root.addWidget(self.canvas, 1)

    @staticmethod
    def _select(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    # ---------------------------------------------------------------- render
    def refresh(self, reset_slice: bool = False) -> None:
        n = self.model.slice_count()
        if reset_slice:
            self.model.set_slice(self.model.default_slice())
        self.model.set_slice(self.model.slice_idx)
        self._slice_guard = True
        self.slice_slider.setRange(0, max(0, n - 1))
        self.slice_slider.setValue(self.model.slice_idx)
        self._slice_guard = False
        self.slice_text.setText(f"{self.model.slice_idx + 1} / {n}")
        self._render()

    def _render(self) -> None:
        case = self.model.current_case
        runs = [self.run_a.currentData(), self.run_b.currentData()]
        self.figure.clear()
        self._axis_slices = {}
        if self.montage:
            n = self.model.slice_count()
            cols = montage_cols(n)
            per_pane = int(np.ceil(n / cols))
            gs = self.figure.add_gridspec(2 * per_pane, cols, wspace=0.04, hspace=0.24)
            for pane, run_id in enumerate(runs):
                for i in range(n):
                    row = pane * per_pane + i // cols
                    ax = self.figure.add_subplot(gs[row, i % cols])
                    self._draw_tile(ax, case, i, run_id, header=(i == 0))
                    self._axis_slices[ax] = i
            self.figure.subplots_adjust(left=0.01, right=0.995, top=0.95, bottom=0.01)
        else:
            for i, run_id in enumerate(runs):
                ax = self.figure.add_subplot(1, 2, i + 1)
                self._draw_tile(ax, case, self.model.slice_idx, run_id,
                                header=True, scale_bar=True)
                self._axis_slices[ax] = self.model.slice_idx
            self.figure.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02,
                                        wspace=0.06)
        self.canvas.draw_idle()
        self.status.emit(self.title_text())

    def _draw_tile(self, ax, case, z: int, run_id: str | None,
                   header: bool = False, scale_bar: bool = False) -> None:
        image = self.model.channel(case.case_id, "LGE")[:, :, z].T
        ax.imshow(image, cmap="gray", origin="lower", aspect="equal",
                  vmin=0, vmax=1, interpolation="nearest")
        gt = self.model.gt(case)
        pred = self.model.prediction(case.case_id, run_id)
        message = None
        if self.overlay == "GT":
            if gt is not None:
                self._paint_source(ax, gt[:, :, z].T)
            else:
                message = "no ground truth"
        elif self.overlay == "Prediction":
            if pred is not None:
                self._paint_source(ax, pred[:, :, z].T)
            elif run_id is None and gt is not None:
                self._paint_source(ax, gt[:, :, z].T)
                message = "GT only"
            else:
                message = "no prediction"
        elif self.overlay == "Both":
            if gt is not None and pred is not None:
                self._paint_mask(ax, gt[:, :, z].T, self.opacity * 0.55)
                self._paint_contours(ax, pred[:, :, z].T, 1.1)
            elif gt is not None:
                self._paint_source(ax, gt[:, :, z].T)
                message = "GT only"
            else:
                message = "no prediction"
        elif self.overlay == "Error":
            if gt is not None and pred is not None:
                rgba = error_rgba(gt[:, :, z].T, pred[:, :, z].T,
                                  self.visible, self.colors, self.opacity)
                ax.imshow(rgba, origin="lower", aspect="equal", interpolation="nearest")
            else:
                message = "error map needs GT + prediction"
        if message and not self.montage:
            h, w = image.shape
            ax.text(w / 2, h * 0.025, message, ha="center", va="bottom",
                    fontsize=7.5, color=qt_theme.TEXT,
                    bbox=dict(facecolor="#0b0e14", edgecolor="none",
                              alpha=0.72, pad=3.0))
        if self.montage:
            ax.set_title(str(z + 1), fontsize=7, color=qt_theme.MUTED, pad=2)
            if z == self.model.slice_idx:
                h, w = image.shape
                ax.add_patch(Rectangle((0, 0), w - 1, h - 1, fill=False,
                                       edgecolor=qt_theme.ACCENT, linewidth=2.0))
        if header:
            title = self._pane_title(case, run_id)
            ax.set_title(title, fontsize=9 if not self.montage else 8,
                         color=qt_theme.TEXT, loc="left", pad=6)
        if scale_bar:
            self._scale_bar(ax, case, image.shape)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _pane_title(self, case, run_id: str | None) -> str:
        name = self.model.run_name(run_id)
        if run_id is None:
            return f"{name} · GT {LABEL_NAMES[self.focus_label]}"
        row = self.model.case_scores(case.case_id, run_id).get(self.focus_label, {})
        dice = row.get("dice")
        suffix = "—" if dice is None else f"{float(dice):.3f}"
        return f"{name} · {LABEL_NAMES[self.focus_label]} Dice {suffix}"

    def _paint_source(self, ax, mask: np.ndarray) -> None:
        if self.style in ("Mask", "Both"):
            self._paint_mask(ax, mask, self.opacity)
        if self.style in ("Contour", "Both"):
            self._paint_contours(ax, mask)

    def _paint_mask(self, ax, mask: np.ndarray, alpha: float) -> None:
        rgba = label_rgba(mask, self.visible, self.colors, alpha)
        ax.imshow(rgba, origin="lower", aspect="equal", interpolation="nearest")

    def _paint_contours(self, ax, mask: np.ndarray, linewidth: float = 1.1) -> None:
        for label in sorted(self.visible):
            binary = (mask == label).astype(float)
            if binary.any():
                ax.contour(binary, levels=[0.5],
                           colors=[contour_color(self.colors[label])],
                           linewidths=linewidth)

    def _scale_bar(self, ax, case, shape) -> None:
        try:
            px_per_mm = 1.0 / float(case.zooms[0])
        except (TypeError, IndexError, ZeroDivisionError):
            return
        height, width = shape
        length_px = 20 * px_per_mm if 20 * px_per_mm <= width * 0.3 else 10 * px_per_mm
        x1, y = width * 0.94, height * 0.055
        ax.plot([x1 - length_px, x1], [y, y], color=qt_theme.TEXT, lw=2)
        ax.text(x1, y * 1.8, f"{int(round(length_px / px_per_mm))} mm",
                color=qt_theme.TEXT, fontsize=7, ha="right", va="bottom")

    # ------------------------------------------------------------- callbacks
    def _on_overlay(self, name: str) -> None:
        self.overlay = name
        self._render()

    def _on_style(self, name: str) -> None:
        self.style = name
        self._render()

    def _on_focus(self, _index: int) -> None:
        self.focus_label = self.focus_combo.currentData()
        self._render()

    def _on_classes(self) -> None:
        self.visible = self.class_toggles.visible()
        self._render()

    def _on_opacity(self, value: int) -> None:
        self.opacity = value / 100.0
        self._render()

    def _on_layout(self, name: str) -> None:
        self.montage = name == "Montage"
        self._render()

    def _on_slice(self, value: int) -> None:
        if self._slice_guard:
            return
        self.model.set_slice(value)
        self.slice_text.setText(f"{self.model.slice_idx + 1} / {self.model.slice_count()}")
        self._render()

    def _on_canvas_click(self, event) -> None:
        if not self.montage or event.inaxes is None:
            return
        index = self._axis_slices.get(event.inaxes)
        if index is None or index == self.model.slice_idx:
            return
        self.model.set_slice(index)
        self._slice_guard = True
        self.slice_slider.setValue(index)
        self._slice_guard = False
        self.slice_text.setText(f"{index + 1} / {self.model.slice_count()}")
        self._render()

    # ------------------------------------------------------------ navigation
    def case_step(self, delta: int) -> None:
        self.model.step_case(delta)
        self.refresh(reset_slice=True)

    def slice_step(self, delta: int) -> None:
        self.model.step_slice(delta)
        self._slice_guard = True
        self.slice_slider.setValue(self.model.slice_idx)
        self._slice_guard = False
        self.slice_text.setText(f"{self.model.slice_idx + 1} / {self.model.slice_count()}")
        self._render()

    def cycle_style(self) -> None:
        order = ["Mask", "Contour", "Both"]
        self.style = order[(order.index(self.style) + 1) % len(order)]
        self.style_control.set_value(self.style)
        self._render()

    def title_text(self) -> str:
        case = self.model.current_case
        return (f"{case.case_id} · compare · slice {self.model.slice_idx + 1}"
                f"/{self.model.slice_count()}")

    def export_png(self, path: str | None = None) -> str:
        self.canvas.draw()
        out = Path(path) if path else repo_path(
            f"var/results/viewer_compare_{self.model.current_case.case_id}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.grab().save(str(out))
        self.status.emit(f"Saved {out}")
        return str(out)
