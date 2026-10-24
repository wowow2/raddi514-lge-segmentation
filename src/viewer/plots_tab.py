"""Plots tab: per-case relationships between scores and case covariates."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.config import EVAL_LABELS, LABEL_NAMES, repo_path

from . import qt_theme
from .model import ViewerModel
from .theme import DARK
from .widgets import Card, Field, SegmentedControl

# Display name -> (clinical meta field | None for the per-row GT volume)
X_OPTIONS = (
    ("GT volume", None),
    ("Age", "Age"),
    ("LVEF", "FEVG"),
    ("Troponin", "Troponin"),
    ("NT-proBNP", "NTProBNP"),
    ("Killip", "Killip Max"),
)


def _number(text) -> float | None:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", str(text).replace(",", ""))
    return float(match.group()) if match else None


class PlotsTab(QWidget):
    """Scatter of per-case Dice vs case covariates, plus box/hist panels."""

    status = pyqtSignal(str)

    def __init__(self, model: ViewerModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.model = model
        self.focus_label = 3
        self.group = "All"
        self.threshold = 0.2
        self.x_name = X_OPTIONS[0][0]
        self._timer_current = None
        self._build()
        self._select_defaults()
        self._replot()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(9)

        controls = QHBoxLayout()
        controls.setSpacing(14)

        self.class_control = SegmentedControl(
            [LABEL_NAMES[i] for i in EVAL_LABELS], LABEL_NAMES[self.focus_label])
        self.class_control.changed.connect(self._on_class)
        controls.addWidget(Field("Class", self.class_control))

        self.x_combo = QComboBox()
        for name, _field in X_OPTIONS:
            self.x_combo.addItem(name)
        self.x_combo.setCurrentText(self.x_name)
        self.x_combo.currentTextChanged.connect(self._on_x)
        controls.addWidget(Field("Scatter X", self.x_combo))

        self.log_check = QCheckBox("log X")
        self.log_check.setChecked(True)
        self.log_check.toggled.connect(self._replot)
        controls.addWidget(self.log_check)

        self.group_combo = QComboBox()
        self.group_combo.addItems(["All", "Pathological", "Normal"])
        self.group_combo.currentTextChanged.connect(self._on_group)
        controls.addWidget(Field("Group", self.group_combo))

        from PyQt5.QtWidgets import QDoubleSpinBox
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 0.9)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(self.threshold)
        self.threshold_spin.valueChanged.connect(self._on_threshold)
        controls.addWidget(Field("Dice below", self.threshold_spin))

        controls.addStretch(1)
        self.note = QLabel("Saved metrics only · runs share the X axis")
        self.note.setObjectName("Muted")
        controls.addWidget(self.note)
        root.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)
        self.figure = Figure(figsize=(9.4, 7.4), dpi=100, facecolor=DARK.figure_face)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setStyleSheet("background: transparent;")
        splitter.addWidget(self.canvas)

        runs_card = Card("Runs")
        self.runs_list = QListWidget()
        self.runs_list.setSelectionMode(QListWidget.NoSelection)
        self.runs_list.itemChanged.connect(lambda _item: self._defer_replot())
        runs_card.add(self.runs_list)
        buttons = QHBoxLayout()
        all_button = QPushButton("All")
        none_button = QPushButton("None")
        for widget, cb in ((all_button, self._select_all), (none_button, self._select_none)):
            widget.setObjectName("Ghost")
            widget.clicked.connect(cb)
            buttons.addWidget(widget)
        buttons.addStretch(1)
        runs_card.body.addLayout(buttons)
        splitter.addWidget(runs_card)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter, 1)

    def _select_all(self) -> None:
        for row in range(self.runs_list.count()):
            self.runs_list.item(row).setCheckState(Qt.Checked)

    def _select_none(self) -> None:
        for row in range(self.runs_list.count()):
            self.runs_list.item(row).setCheckState(Qt.Unchecked)

    def _select_defaults(self) -> None:
        preferred = (
            "nnUNet_validation_pp_grow1",
            "nnUNet_validation_500over_pp_grow1",
            "nnUNet_validation_pp",
            "nnUNet_validation",
            "nnUNet_validation_501_pp",
            "polar_preds_add_t0.7",
        )
        ids = self.model.metric_run_ids()
        chosen = [i for i in preferred if i in ids]
        for run_id in ids:
            if run_id not in chosen and len(chosen) < 4:
                chosen.append(run_id)
        self.runs_list.blockSignals(True)
        for run_id in ids:
            item = QListWidgetItem(self.model.run_name(run_id))
            item.setData(Qt.UserRole, run_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if run_id in chosen else Qt.Unchecked)
            self.runs_list.addItem(item)
        self.runs_list.blockSignals(False)

    def selected_runs(self) -> list[str]:
        return [self.runs_list.item(r).data(Qt.UserRole)
                for r in range(self.runs_list.count())
                if self.runs_list.item(r).checkState() == Qt.Checked]

    # ------------------------------------------------------------ state sync
    def refresh(self) -> None:
        pass  # plots only depend on saved metrics; nothing to rescan

    def _on_class(self, name: str) -> None:
        for label in EVAL_LABELS:
            if LABEL_NAMES[label] == name:
                self.focus_label = label
        self._replot()

    def _on_x(self, name: str) -> None:
        self.x_name = name
        self._replot()

    def _on_group(self, group: str) -> None:
        self.group = group
        self._replot()

    def _on_threshold(self, value: float) -> None:
        self.threshold = value
        self._replot()

    def _defer_replot(self) -> None:
        self._replot()

    # ----------------------------------------------------------------- plots
    def _rows(self, run_id: str) -> list[dict]:
        table = self.model.run_table(run_id)
        if table is None:
            return []
        label = self.focus_label
        out = []
        for row in table.rows:
            if row.get("class_id") != label or not row.get("dice_defined"):
                continue
            if self.group != "All" and row.get("group") != self.group:
                continue
            value = row.get("dice")
            if value is None or not np.isfinite(value):
                continue
            out.append(row)
        return out

    def _x_values(self, rows: list[dict]) -> dict[str, float]:
        field = dict(X_OPTIONS).get(self.x_name)
        values = {}
        for row in rows:
            if field is None:
                vox = row.get("gt_voxels")
                if vox is not None and np.isfinite(vox) and vox > 0:
                    values[row["case_id"]] = float(vox)
                continue
            case = self.model.case_by_id(row["case_id"])
            number = _number(case.clinical_meta.get(field)) if case else None
            if number is not None:
                values[row["case_id"]] = number
        return values

    def _replot(self) -> None:
        runs = self.selected_runs()
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2, width_ratios=(1.5, 1.0),
                                      hspace=0.34, wspace=0.22)
        ax_sc = self.figure.add_subplot(gs[:, 0])
        ax_box = self.figure.add_subplot(gs[0, 1])
        ax_hist = self.figure.add_subplot(gs[1, 1])

        if not runs:
            for ax in (ax_sc, ax_box, ax_hist):
                ax.text(0.5, 0.5, "select runs on the right", ha="center",
                        va="center", color=qt_theme.MUTED, fontsize=11)
                ax.set_axis_off()
            self.canvas.draw_idle()
            return

        x_values = self._x_values(self._rows(runs[0]))
        use_log = self.log_check.isChecked() and self.x_name == X_OPTIONS[0][0]

        # ---- scatter
        for i, run_id in enumerate(runs):
            rows = self._rows(run_id)
            xs, ys = [], []
            for row in rows:
                x = x_values.get(row["case_id"])
                if x is None or x <= 0:
                    continue
                xs.append(x)
                ys.append(row["dice"])
            if not xs:
                continue
            color = qt_theme.series_color(i)
            ax_sc.scatter(xs, ys, s=20, alpha=0.75, color=color,
                          edgecolors="none",
                          label=f"{self._short(self.model.run_name(run_id))} (n={len(xs)})")
            if len(set(xs)) > 2 and use_log:
                logx = np.log10(xs)
                slope, intercept = np.polyfit(logx, ys, 1)
                grid = np.geomspace(min(xs), max(xs), 50)
                ax_sc.plot(grid, np.polyval((slope, intercept), np.log10(grid)),
                           color=color, lw=1.3, alpha=0.85)

        ax_sc.set_xlabel(f"{self.x_name}{' (voxels)' if use_log else ''}")
        ax_sc.set_ylabel(f"{LABEL_NAMES[self.focus_label]} Dice")
        title = f"Per-case {LABEL_NAMES[self.focus_label].lower()} Dice vs {self.x_name.lower()}"
        ax_sc.set_title(title, color=qt_theme.TEXT, fontsize=10.5)
        if use_log:
            ax_sc.set_xscale("log")
        ax_sc.grid(alpha=0.18, lw=0.6)
        if len(runs) <= 6:
            ax_sc.legend(fontsize=6.7, loc="lower left", framealpha=0.0,
                         labelcolor="#c9d1e0")

        # ---- per-group box per run (group All is trivially one box)
        for i, run_id in enumerate(runs):
            rows = self._rows(run_id)
            groups = []
            for group in ("Pathological", "Normal"):
                vals = [r["dice"] for r in rows if r.get("group") == group]
                groups.append((group, vals))
            xs, ys, widths = [], [], []
            for j, (_g, vals) in enumerate(groups):
                if vals:
                    xs.append(i - 0.15 + 0.3 * j)
                    ys.append(vals)
                    widths.append(0.24)
            if ys:
                bp = ax_box.boxplot(ys, positions=xs, widths=widths,
                                    showfliers=False, patch_artist=True,
                                    medianprops={"color": "#e6e9f0", "lw": 1.0},
                                    whiskerprops={"color": qt_theme.MUTED, "lw": 0.7},
                                    capprops={"color": qt_theme.MUTED, "lw": 0.7})
                for box in bp["boxes"]:
                    box.set(facecolor=qt_theme.series_color(i), alpha=0.35, lw=0.8)
        ax_box.set_xticks(range(len(runs)))
        ax_box.set_xticklabels([self._short(self.model.run_name(r), 12) for r in runs],
                               rotation=40, ha="right", fontsize=6.5)
        ax_box.set_title("Dice by group", fontsize=9, color="#c9d1e0")
        ax_box.set_ylim(-0.02, 1.02)
        ax_box.tick_params(labelsize=7)
        ax_box.grid(axis="y", alpha=0.18, lw=0.6)

        # ---- cumulative dice histogram (first selected run)
        vals = sorted(r["dice"] for r in self._rows(runs[0]))
        if vals:
            n = len(vals)
            bins = np.linspace(0, 1, 26)
            y = 100 * np.searchsorted(np.asarray(vals), bins, side="right") / n
            ax_hist.fill_between(bins, y, color=qt_theme.series_color(0),
                                 alpha=0.35, step="mid")
            ax_hist.plot(bins, y, color=qt_theme.series_color(0), lw=1.2,
                         drawstyle="steps-mid")
            below = 100 * np.searchsorted(vals, self.threshold) / n if vals else 0
            ax_hist.axvline(self.threshold, color="#ff6b6b", lw=1.0, ls="--")
            ax_hist.text(self.threshold + 0.02, 8, f"{below:.0f}% < {self.threshold:.2f}",
                         fontsize=7, color="#ff6b6b")
        ax_hist.set_title(f"Per-case Dice · {self._short(self.model.run_name(runs[0]), 18)}",
                          fontsize=9, color="#c9d1e0")
        ax_hist.set_ylim(0, 104)
        ax_hist.set_xlabel("Dice", fontsize=7.5)
        ax_hist.set_ylabel("% of cases", fontsize=7.5)
        ax_hist.tick_params(labelsize=7)
        ax_hist.grid(alpha=0.18, lw=0.6)

        self.canvas.draw_idle()

    @staticmethod
    def _short(name: str, width: int = 20) -> str:
        name = name.replace("nnUNet_validation_", "").replace("nnUNet_", "")
        return name if len(name) <= width else name[: width - 1] + "…"

    # ---------------------------------------------------------------- export
    def export_png(self, path: str | None = None) -> str:
        out = Path(path) if path else repo_path(
            f"var/results/viewer_plots_{LABEL_NAMES[self.focus_label].lower()}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(str(out), dpi=200, facecolor=DARK.figure_face,
                            bbox_inches="tight")
        self.status.emit(f"Saved {out}")
        return str(out)
