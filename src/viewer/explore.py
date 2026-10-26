"""Explore tab: one case, overlays, masks/contours, whole-volume montage."""

from __future__ import annotations

import html
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
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.config import LABEL_NAMES, LABEL_SCAR, DEFAULT_REPORT_DIR

from . import qt_theme
from .model import (CHANNELS, ViewerModel, contour_color, error_rgba, label_rgba,
                    montage_cols, probability_rgba)
from .theme import DARK
from .widgets import ClassToggles, Field, SegmentedControl

OVERLAYS = ("Prediction", "GT", "Both", "Error", "Probability")


class ExploreTab(QWidget):
    """Case-centric view with a montage/single-slice layout."""

    status = pyqtSignal(str)

    def __init__(self, model: ViewerModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.model = model
        self.colors = {i: DARK.classes[i - 1][1] for i in range(1, 5)}
        self.visible = {1, 2, 3, 4}
        self.focus_label = LABEL_SCAR
        self.overlay = "Prediction"
        self.style = "Mask"
        self.channel = "LGE"
        self.opacity = 0.70
        self.montage = False
        self.run_id = model.default_pred_run()
        if self.run_id is None:
            self.overlay = "GT"
        self.prob_source = model.default_prob_source(self.run_id)
        self._axis_slices: dict[object, int] = {}
        self._slice_guard = False

        self._build()
        self.model.set_case_index(self.model.case_pos, reset_slice=True)
        self.refresh(reset_slice=True)

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(9)

        row1 = QHBoxLayout()
        row1.setSpacing(14)
        self.run_combo = QComboBox()
        self.run_combo.addItem("GT only", None)
        for choice in self.model.run_choices(only_preds=True):
            self.run_combo.addItem(choice.name, choice.run_id)
        self._select_combo(self.run_combo, self.run_id)
        self.run_combo.currentIndexChanged.connect(self._on_run)
        row1.addWidget(Field("Run", self.run_combo))

        self.overlay_combo = QComboBox()
        for name in OVERLAYS:
            self.overlay_combo.addItem(name, name)
        self.overlay_combo.setCurrentText(self.overlay)
        self.overlay_combo.currentTextChanged.connect(self._on_overlay)
        row1.addWidget(Field("Overlay", self.overlay_combo))

        self.prob_combo = QComboBox()
        for label in self.model.prob_sources:
            self.prob_combo.addItem(label, label)
        self._select_combo(self.prob_combo, self.prob_source)
        self.prob_combo.currentIndexChanged.connect(self._on_prob_source)
        self.prob_field = Field("Source", self.prob_combo)
        self.prob_field.setVisible(False)
        row1.addWidget(self.prob_field)

        self.style_control = SegmentedControl(["Mask", "Contour", "Both"], self.style)
        self.style_control.changed.connect(self._on_style)
        row1.addWidget(Field("Style", self.style_control))

        self.channel_combo = QComboBox()
        for name in CHANNELS:
            self.channel_combo.addItem(name, name)
        self.channel_combo.currentTextChanged.connect(self._on_channel)
        row1.addWidget(Field("View", self.channel_combo))

        self.layout_control = SegmentedControl(["Single", "Montage"], "Single")
        self.layout_control.changed.connect(self._on_layout)
        row1.addWidget(Field("Layout", self.layout_control))
        row1.addStretch(1)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(14)
        self.focus_combo = QComboBox()
        for label in (1, 2, 3, 4):
            self.focus_combo.addItem(LABEL_NAMES[label], label)
        self.focus_combo.setCurrentIndex(2)
        self.focus_combo.currentIndexChanged.connect(self._on_focus)
        row2.addWidget(Field("Focus", self.focus_combo))

        self.class_toggles = ClassToggles()
        self.class_toggles.changed.connect(self._on_classes)
        row2.addWidget(self.class_toggles)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(int(self.opacity * 100))
        self.opacity_slider.setFixedWidth(110)
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        row2.addWidget(Field("Opacity", self.opacity_slider))

        row2.addStretch(1)
        self.prev_button = QPushButton("‹  Previous")
        self.prev_button.setObjectName("Ghost")
        self.prev_button.clicked.connect(lambda: self._step_case(-1))
        self.next_button = QPushButton("Next  ›")
        self.next_button.setObjectName("Ghost")
        self.next_button.clicked.connect(lambda: self._step_case(1))
        row2.addWidget(self.prev_button)
        row2.addWidget(self.next_button)
        self.queue_label = QLabel("")
        self.queue_label.setObjectName("ChipLabel")
        self.queue_label.hide()
        row2.addWidget(self.queue_label)

        self.queue_exit = QPushButton("Exit queue")
        self.queue_exit.setObjectName("Ghost")
        self.queue_exit.hide()
        self.queue_exit.clicked.connect(self._exit_queue)
        row2.addWidget(self.queue_exit)
        root.addLayout(row2)

        slice_row = QHBoxLayout()
        slice_row.setSpacing(10)
        slice_label = QLabel("Slice")
        slice_label.setObjectName("Muted")
        slice_row.addWidget(slice_label)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self._on_slice)
        slice_row.addWidget(self.slice_slider, 1)
        self.slice_text = QLabel("1 / 1")
        self.slice_text.setObjectName("Muted")
        self.slice_text.setMinimumWidth(56)
        slice_row.addWidget(self.slice_text)
        self.legend_label = QLabel("")
        self.legend_label.setObjectName("Muted")
        slice_row.addWidget(self.legend_label)
        root.addLayout(slice_row)

        splitter = QSplitter(Qt.Horizontal)
        self.figure = Figure(figsize=(7.2, 7.2), dpi=100, facecolor=DARK.figure_face)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setStyleSheet("background: transparent;")
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        splitter.addWidget(self.canvas)

        self.info = QTextBrowser()
        self.info.setOpenExternalLinks(False)
        self.info.setMinimumWidth(320)
        self.info.setMaximumWidth(430)
        splitter.addWidget(self.info)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter, 1)

        self._refresh_run_choices()

    # ------------------------------------------------------------- state sync
    def _refresh_run_choices(self) -> None:
        current = self.run_combo.currentData()
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        self.run_combo.addItem("GT only", None)
        for choice in self.model.run_choices(only_preds=True):
            self.run_combo.addItem(choice.name, choice.run_id)
        self.run_combo.blockSignals(False)
        if current in {c.run_id for c in self.model.run_choices(only_preds=True)}:
            self._select_combo(self.run_combo, current)
        elif self.run_combo.count() > 1:
            self.run_combo.setCurrentIndex(1)
            self.run_id = self.run_combo.currentData()
        else:
            self.run_combo.setCurrentIndex(0)
            self.run_id = None

    @staticmethod
    def _select_combo(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def refresh(self, reset_slice: bool = False) -> None:
        case = self.model.current_case
        n = self.model.slice_count()
        if reset_slice:
            self.model.set_slice(self.model.default_slice(case))
        self.model.set_slice(self.model.slice_idx)
        self._slice_guard = True
        self.slice_slider.setRange(0, max(0, n - 1))
        self.slice_slider.setValue(self.model.slice_idx)
        self._slice_guard = False
        self.slice_text.setText(f"{self.model.slice_idx + 1} / {n}")
        self._render()
        self._update_hud()
        self._update_queue_chip()

    def _render(self) -> None:
        case = self.model.current_case
        z = self.model.slice_idx
        self.figure.clear()
        self._axis_slices = {}
        if self.montage:
            n = self.model.slice_count()
            cols = montage_cols(n)
            rows = int(np.ceil(n / cols))
            gs = self.figure.add_gridspec(rows, cols, wspace=0.04, hspace=0.16)
            for i in range(n):
                ax = self.figure.add_subplot(gs[i // cols, i % cols])
                self._draw_tile(ax, case, i)
                self._axis_slices[ax] = i
            self.figure.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.01)
        else:
            ax = self.figure.add_subplot(111)
            self._draw_tile(ax, case, z, scale_bar=True)
            self._axis_slices[ax] = z
            self._paint_gate_marks(ax, case, z)
            self.figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        self.canvas.draw_idle()
        self.status.emit(self.title_text())

    def _draw_tile(self, ax, case, z: int, scale_bar: bool = False) -> None:
        image = self.model.channel(case.case_id, self.channel)[:, :, z].T
        ax.imshow(image, cmap="gray", origin="lower", aspect="equal",
                  vmin=0, vmax=1, interpolation="nearest")
        gt = self.model.gt(case)
        pred = self.model.prediction(case.case_id, self.run_id)
        message = None
        if self.overlay == "GT":
            if gt is not None:
                self._paint_source(ax, gt[:, :, z].T)
            else:
                message = "no ground truth"
        elif self.overlay == "Prediction":
            if pred is not None:
                self._paint_source(ax, pred[:, :, z].T)
            else:
                message = "no prediction"
        elif self.overlay == "Both":
            if gt is not None and pred is not None:
                self._paint_mask(ax, gt[:, :, z].T, alpha=self.opacity * 0.55)
                self._paint_contours(ax, pred[:, :, z].T, linewidth=1.1)
            elif pred is not None:
                self._paint_source(ax, pred[:, :, z].T)
                message = "no ground truth"
            else:
                message = "no prediction"
        elif self.overlay == "Error":
            if gt is not None and pred is not None:
                rgba = error_rgba(gt[:, :, z].T, pred[:, :, z].T,
                                  self.visible, self.colors, self.opacity)
                ax.imshow(rgba, origin="lower", aspect="equal", interpolation="nearest")
            else:
                message = "error map needs GT + prediction"
        elif self.overlay == "Probability":
            prob = None
            if self.prob_source:
                prob = self.model.probability(case.case_id, self.prob_source,
                                              self.focus_label)
            if prob is not None:
                color = self.colors[self.focus_label][:3]
                slice_prob = prob[:, :, z].T
                rgba = probability_rgba(slice_prob, color, self.opacity)
                ax.imshow(rgba, origin="lower", aspect="equal", interpolation="bilinear")
                for threshold, dashed in ((0.2, True), (0.5, False)):
                    if (slice_prob > threshold).any():
                        ax.contour((slice_prob > threshold).astype(float),
                                   levels=[0.5], colors=[color],
                                   linewidths=1.0, linestyles="dashed" if dashed else "solid")
                if slice_prob.max() < 0.05:
                    message = f"low confidence · max P = {slice_prob.max():.3f}"
            else:
                message = "no probability map"

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
        else:
            self._scale_bar(ax, case, image.shape)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _paint_source(self, ax, mask: np.ndarray) -> None:
        if self.style in ("Mask", "Both"):
            self._paint_mask(ax, mask, alpha=self.opacity)
        if self.style in ("Contour", "Both"):
            self._paint_contours(ax, mask)

    def _paint_mask(self, ax, mask: np.ndarray, alpha: float) -> None:
        rgba = label_rgba(mask, self.visible, self.colors, alpha)
        ax.imshow(rgba, origin="lower", aspect="equal", interpolation="nearest")

    def _paint_contours(self, ax, mask: np.ndarray, linewidth: float = 1.2) -> None:
        for label in sorted(self.visible):
            shifted = (mask == label).astype(float)
            if shifted.any():
                ax.contour(shifted, levels=[0.5],
                           colors=[contour_color(self.colors[label])],
                           linewidths=linewidth)

    def _paint_gate_marks(self, ax, case, z: int) -> None:
        """Dashed outlines around prediction components the challenge gates
        would remove (single view, overlay modes that show the prediction)."""
        if self.overlay == "GT":
            return
        pred = self.model.prediction(case.case_id, self.run_id)
        if pred is None:
            return
        info = self.model.gate_preview(case.case_id, self.run_id)
        if not info or not info["changed"]:
            return
        for label in (3, 4):
            mask = (info["removed"] & (pred == label))[:, :, z].astype(float)
            if mask.any():
                ax.contour(mask, levels=[0.5], colors=["#f2f4f8"],
                           linewidths=1.1, linestyles="dashed")

    def _scale_bar(self, ax, case, shape) -> None:
        try:
            px_per_mm = 1.0 / float(case.zooms[0])
        except (TypeError, IndexError, ZeroDivisionError):
            return
        height, width = shape
        length_mm = 20 if 20 * px_per_mm <= width * 0.35 else 10
        length_px = length_mm * px_per_mm
        x1 = width * 0.94
        y = height * 0.055
        ax.plot([x1 - length_px, x1], [y, y], color=qt_theme.TEXT, lw=2,
                solid_capstyle="butt")
        ax.text(x1, y * 1.8, f"{length_mm} mm", color=qt_theme.TEXT,
                fontsize=7, ha="right", va="bottom")

    # ------------------------------------------------------------------- HUD
    def _update_hud(self) -> None:
        case = self.model.current_case
        meta = case.clinical_meta or {}
        pred = self.model.prediction(case.case_id, self.run_id)
        scores = self.model.case_scores(case.case_id, self.run_id)
        vols = case.voxel_counts

        def esc(value) -> str:
            return html.escape(str(value))

        def cls_color(lbl: int) -> str:
            r, g, b = (round(255 * c) for c in self.colors[lbl][:3])
            return f"rgb({r},{g},{b})"

        parts = [
            "<div style='font-size:10pt; line-height:1.45'>",
            f"<div style='font-size:15pt; font-weight:700'>{esc(case.case_id)}"
            f" <span style='font-size:9pt; color:{qt_theme.MUTED}'>"
            f"{esc(case.group)}</span></div>",
            f"<div style='color:{qt_theme.MUTED}; font-size:8.5pt'>"
            f"Case {self.model.case_pos + 1}/{self.model.case_count()}"
            f" &nbsp;·&nbsp; Slice {self.model.slice_idx + 1}/{self.model.slice_count()}"
            f" &nbsp;·&nbsp; Fold {self.model.fold_of(case.case_id, self.run_id) if self.run_id else '—'}"
            "</div>",
        ]

        if self.overlay == "Error":
            parts.append(
                f"<div style='margin-top:8px; color:{qt_theme.MUTED}; font-size:8.5pt'>"
                f"<span style='color:#ff9433'>■</span> false positive"
                f" &nbsp; <span style='color:#19c7ff'>■</span> false negative"
                f" &nbsp; <span style='color:#5bc8ff'>■</span>"
                f"<span style='color:#3ddc84'>■</span>"
                f"<span style='color:#ff4040'>■</span>"
                f"<span style='color:#ffe014'>■</span> true positive</div>")
        elif self.overlay == "Probability" and self.prob_source:
            parts.append(
                f"<div style='margin-top:8px; color:{qt_theme.MUTED}; font-size:8.5pt'>"
                f"P({esc(LABEL_NAMES[self.focus_label])}) · source: {esc(self.prob_source)}</div>")
            prob = self.model.probability(case.case_id, self.prob_source, self.focus_label)
            if prob is not None:
                z = self.model.slice_idx
                p_case = prob
                p_slice = prob[:, :, z]
                parts.append(self._kv_table([
                    ("Case max", f"{float(p_case.max()):.3f}"),
                    ("Slice max", f"{float(p_slice.max()):.3f}"),
                    ("Voxels > 0.2", f"{int((p_case > 0.2).sum()):,}"),
                    ("Voxels > 0.5", f"{int((p_case > 0.5).sum()):,}"),
                ]))

        parts.append("<div style='height:1px; background:%s; margin:10px 0'></div>" % qt_theme.BORDER)

        # Patient
        parts.append(self._section("Patient"))
        patient_rows = [
            ("Age / Sex", f"{esc(meta.get('Age', '—'))} / {esc(meta.get('Sex', '—'))}"),
            ("LVEF", f"{esc(meta.get('FEVG', '—'))} %"),
            ("Troponin", esc(meta.get("Troponin", "—"))),
            ("NT-proBNP", esc(meta.get("NTProBNP", "—"))),
            ("Killip max", esc(meta.get("Killip Max", "—"))),
            ("Risk", " · ".join(filter(None, [
                f"HTN {esc(meta.get('Arterial hypertension'))}" if meta.get("Arterial hypertension") not in (None, "", "0") else "",
                f"DM {esc(meta.get('Diabetes'))}" if meta.get("Diabetes") not in (None, "", "0") else "",
                f"Smoke {esc(meta.get('Tobacco'))}" if meta.get("Tobacco") not in (None, "", "0") else "",
            ])) or "—"),
        ]
        parts.append(self._kv_table(patient_rows))

        # Geometry + GT
        parts.append(self._section("Geometry & ground truth"))
        rows = [
            ("Matrix", " × ".join(str(v) for v in case.shape or [])),
            ("Spacing", " × ".join(str(v) for v in case.zooms or []) + " mm"),
            ("Slices", str(self.model.slice_count())),
        ]
        for lbl in (1, 2, 3, 4):
            if not case.has_gt:
                rows.append((LABEL_NAMES[lbl], "— (blind)"))
                continue
            extra = "present" if vols.get(lbl, 0) else "absent"
            rows.append((LABEL_NAMES[lbl],
                         f"{vols.get(lbl, 0):,} vox · {extra}"))
        parts.append(self._kv_table(rows))

        # Predictions
        parts.append(self._section("Prediction"))
        if self.run_id is None:
            parts.append(f"<div style='color:{qt_theme.MUTED}'>No run selected (GT only).</div>")
        elif pred is None:
            parts.append(f"<div style='color:{qt_theme.MUTED}'>No prediction file for this case.</div>")
        else:
            prow = []
            for lbl in (1, 2, 3, 4):
                n_pred = int((pred == lbl).sum())
                prow.append((LABEL_NAMES[lbl], f"{n_pred:,} vox"))
            parts.append(self._kv_table(prow))

            table = [f"<table width='100%' cellspacing='0' cellpadding='3' "
                     f"style='font-size:8.7pt;'>"
                     f"<tr style='color:{qt_theme.MUTED}'>"
                     f"<th align='left'>Class</th><th align='right'>Dice</th>"
                     f"<th align='right'>HD95</th><th align='right'>Rec</th>"
                     f"<th align='right'>Prec</th></tr>"]
            for lbl in (1, 2, 3, 4):
                row = scores.get(lbl, {})
                dice = row.get("dice")
                hd95 = row.get("hd95")
                tp, fp, fn = row.get("tp"), row.get("fp"), row.get("fn")
                recall = (tp / (tp + fn)) if tp is not None and (tp + fn) else None
                prec = (tp / (tp + fp)) if tp is not None and (tp + fp) else None
                table.append(
                    "<tr>"
                    f"<td><span style='color:{cls_color(lbl)}'>■</span> {LABEL_NAMES[lbl]}</td>"
                    f"<td align='right' style='color:{qt_theme.score_color(dice)}; font-weight:600'>"
                    f"{self._fmt(dice, 3)}</td>"
                    f"<td align='right'>{self._fmt(hd95, 1)}</td>"
                    f"<td align='right'>{self._fmt(recall, 2)}</td>"
                    f"<td align='right'>{self._fmt(prec, 2)}</td>"
                    "</tr>")
            table.append("</table>")
            parts.append("".join(table))

            confusion = self.model.confusion(case.case_id, self.run_id, self.focus_label)
            if confusion:
                parts.append(self._section(
                    f"GT {LABEL_NAMES[self.focus_label]} → prediction"))
                conf_rows = [(name, f"{100 * frac:.0f}%")
                             for name, frac in confusion.items() if frac > 0]
                parts.append(self._kv_table(conf_rows))

        # Challenge gate preview
        if self.run_id and pred is not None:
            info = self.model.gate_preview(case.case_id, self.run_id)
            if info is not None:
                parts.append(self._section("Challenge gate (preview)"))
                rows = []
                if not info["changed"]:
                    rows.append(("Effect", "no change (already gated)"))
                else:
                    if info["scar"]["voxels"]:
                        rows.append(("Scar removed",
                                     f"{info['scar']['voxels']} vox · "
                                     f"{info['scar']['comps']} specks"))
                    if info["mvo"]["voxels"]:
                        rows.append(("MVO removed",
                                     f"{info['mvo']['voxels']} vox · "
                                     f"{info['mvo']['comps']} blobs"))
                    if info["before"][self.focus_label] < info["after"][self.focus_label]:
                        rows.append(("Focus Dice",
                                     f"{info['before'][self.focus_label]:.2f} → "
                                     f"{info['after'][self.focus_label]:.2f}"))
                    if info["before"]["all"] < info["after"]["all"]:
                        rows.append(("Mean Dice",
                                     f"{info['before']['all']:.3f} → "
                                     f"{info['after']['all']:.3f}"))
                parts.append(self._kv_table(rows))

        # Run context
        run = next((r for r in self.model.runs if r.run_id == self.run_id), None)
        if run is not None:
            parts.append(self._section("Run"))
            tags = ", ".join(run.manifest.tags) or "—"
            parts.append(self._kv_table([
                ("ID", esc(run.run_id)),
                ("Stage", esc(run.manifest.stage or "—")),
                ("Tags", esc(tags)),
                ("Metrics", "saved" if scores else "computed"),
            ]))

        parts.append("</div>")
        self.info.setHtml("".join(parts))

    @staticmethod
    def _fmt(value, digits: int) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "—"
        return f"{float(value):.{digits}f}"

    @staticmethod
    def _section(title: str) -> str:
        return (f"<div style='margin:13px 0 4px; font-size:8pt; letter-spacing:0.06em; "
                f"color:{qt_theme.MUTED}; font-weight:600'>{html.escape(title.upper())}</div>")

    @staticmethod
    def _kv_table(rows) -> str:
        body = "".join(
            f"<tr><td style='color:{qt_theme.MUTED}; padding:1px 10px 1px 0'>{k}</td>"
            f"<td style='padding:1px 0' align='right'>{v}</td></tr>"
            for k, v in rows)
        return f"<table width='100%' cellspacing='0' cellpadding='0' style='font-size:8.7pt'>{body}</table>"

    # ------------------------------------------------------------- callbacks
    def _on_run(self, _index: int) -> None:
        self.run_id = self.run_combo.currentData()
        if self.prob_source is None:
            self.prob_source = self.model.default_prob_source(self.run_id)
            self._select_combo(self.prob_combo, self.prob_source)
        self._render()
        self._update_hud()

    def _on_overlay(self, name: str) -> None:
        self.overlay = name
        self.prob_field.setVisible(name == "Probability")
        if name == "Probability" and self.prob_source is None:
            self.prob_source = self.model.default_prob_source(self.run_id)
            self._select_combo(self.prob_combo, self.prob_source)
        self._render()
        self._update_hud()

    def _on_prob_source(self, _index: int) -> None:
        self.prob_source = self.prob_combo.currentData()
        if self.overlay == "Probability":
            self._render()
            self._update_hud()

    def _on_style(self, name: str) -> None:
        self.style = name
        self._render()

    def _on_channel(self, name: str) -> None:
        self.channel = name
        self._render()

    def _on_layout(self, name: str) -> None:
        self.montage = name == "Montage"
        self._render()

    def _on_focus(self, _index: int) -> None:
        self.focus_label = self.focus_combo.currentData()
        if self.overlay == "Probability":
            self._render()
        self._update_hud()

    def _on_classes(self) -> None:
        self.visible = self.class_toggles.visible()
        self._render()

    def _on_opacity(self, value: int) -> None:
        self.opacity = value / 100.0
        self._render()

    def _on_slice(self, value: int) -> None:
        if self._slice_guard:
            return
        self.model.set_slice(value)
        self.slice_text.setText(f"{self.model.slice_idx + 1} / {self.model.slice_count()}")
        self._render()
        self._update_hud()

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
        self._update_hud()

    # ------------------------------------------------------------ navigation
    def _step_case(self, delta: int) -> None:
        if self.model.queue_active() and self.model.queue_step(delta):
            self.refresh(reset_slice=True)
            self.status.emit(self.title_text())
            self._update_queue_chip()
            return
        self.model.step_case(delta)
        self.refresh(reset_slice=True)
        self.status.emit(self.title_text())
        self._update_queue_chip()

    def _exit_queue(self) -> None:
        self.model.queue_clear()
        self._update_queue_chip()
        self.status.emit("queue cleared")

    def _update_queue_chip(self) -> None:
        if not self.model.queue_active() or self.model.current_case is None:
            self.queue_label.hide()
            self.queue_exit.hide()
            return
        ids = self.model.queue_ids()
        pos = ids.index(self.model.current_case.case_id) \
            if self.model.current_case.case_id in ids else 0
        self.queue_label.setText(f"Review queue {pos + 1}/{len(ids)}")
        self.queue_label.show()
        self.queue_exit.show()

    def case_step(self, delta: int) -> None:
        self._step_case(delta)

    def slice_step(self, delta: int) -> None:
        self.model.step_slice(delta)
        self._slice_guard = True
        self.slice_slider.setValue(self.model.slice_idx)
        self._slice_guard = False
        self.slice_text.setText(f"{self.model.slice_idx + 1} / {self.model.slice_count()}")
        self._render()
        self._update_hud()

    def cycle_style(self) -> None:
        order = ["Mask", "Contour", "Both"]
        self.style = order[(order.index(self.style) + 1) % len(order)]
        self.style_control.set_value(self.style)
        self._render()

    def title_text(self) -> str:
        case = self.model.current_case
        run = self.model.run_name(self.run_id)
        return (f"{case.case_id} · {case.group} · slice {self.model.slice_idx + 1}"
                f"/{self.model.slice_count()} · {run} · {self.overlay}/{self.style}")

    def export_png(self, path: str | None = None) -> str:
        self.canvas.draw()
        out = Path(path) if path else Path(DEFAULT_REPORT_DIR) / (
            f"viewer_{self.model.current_case.case_id}"
            f"_s{self.model.slice_idx + 1}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.grab().save(str(out))
        self.status.emit(f"Saved {out}")
        return str(out)
