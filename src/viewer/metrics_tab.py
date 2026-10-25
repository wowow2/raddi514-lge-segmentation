"""Metrics tab: distributions, scorecard, and failure triage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QComboBox,
    QHeaderView,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.config import EVAL_LABELS, LABEL_NAMES, repo_path

from . import qt_theme
from .model import ViewerModel, bootstrap_ci
from .theme import DARK
from .widgets import Card, Field, SegmentedControl

GROUP_COLORS = {"Pathological": "#4aa3ff", "Normal": "#ff6b6b", "Unknown": "#9aa5b1"}


class MetricsTab(QWidget):
    """Cross-run metric overview that can jump back to a case."""

    case_requested = pyqtSignal(str)
    queue_requested = pyqtSignal(list)
    status = pyqtSignal(str)

    def __init__(self, model: ViewerModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.model = model
        self.metric_name = "Dice"
        self.group = "All"
        self.threshold = 0.2
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._replot)
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
        self.metric_control = SegmentedControl(["Dice", "HD95"], self.metric_name)
        self.metric_control.changed.connect(self._on_metric)
        controls.addWidget(Field("Metric", self.metric_control))

        self.group_combo = QComboBox()
        self.group_combo.addItems(["All", "Pathological", "Normal"])
        self.group_combo.currentTextChanged.connect(self._on_group)
        controls.addWidget(Field("Group", self.group_combo))

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 0.9)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(self.threshold)
        self.threshold_spin.valueChanged.connect(self._on_threshold)
        controls.addWidget(Field("Dice below", self.threshold_spin))
        self.queue_button = QPushButton("Review failures")
        self.queue_button.setObjectName("Ghost")
        self.queue_button.clicked.connect(self._review_failures)
        controls.addWidget(self.queue_button)

        controls.addStretch(1)
        self.note = QLabel("Plots use saved metrics only · no volumes are loaded")
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

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(10, 0, 0, 0)
        side_layout.setSpacing(9)

        runs_card = Card("Runs")
        self.runs_list = QListWidget()
        self.runs_list.setAlternatingRowColors(True)
        self.runs_list.setMaximumHeight(150)
        self.runs_list.itemChanged.connect(lambda _item: self._timer.start())
        runs_card.add(self.runs_list)
        buttons = QHBoxLayout()
        all_button = QPushButton("All")
        all_button.setObjectName("Ghost")
        all_button.clicked.connect(lambda: self._set_all(True))
        none_button = QPushButton("None")
        none_button.setObjectName("Ghost")
        none_button.clicked.connect(lambda: self._set_all(False))
        buttons.addWidget(all_button)
        buttons.addWidget(none_button)
        buttons.addStretch(1)
        runs_card.body.addLayout(buttons)
        runs_card.setMaximumHeight(225)
        side_layout.addWidget(runs_card, 0)

        score_card = Card("Scorecard")
        self.score_table = QTableWidget(0, 6)
        self.score_table.setHorizontalHeaderLabels(
            ["Run", "Cavity", "Myo", "Scar", "MVO", "All"])
        self.score_table.verticalHeader().setVisible(False)
        self.score_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.score_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.score_table.setAlternatingRowColors(True)
        self.score_table.horizontalHeader().setStretchLastSection(True)
        self.score_table.setMinimumHeight(150)
        self.score_table.verticalHeader().setDefaultSectionSize(26)
        self.score_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.score_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        score_card.add(self.score_table)
        side_layout.addWidget(score_card, 1)

        bad_card = Card("Failure triage")
        self.hint = QLabel("Double-click a case to open it in Explore")
        self.hint.setObjectName("Muted")
        bad_card.add(self.hint)
        self.bad_table = QTableWidget(0, 7)
        self.bad_table.setHorizontalHeaderLabels(
            ["Case", "Group", "Fold", "Cavity", "Myo", "Scar", "MVO"])
        self.bad_table.verticalHeader().setVisible(False)
        self.bad_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.bad_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.bad_table.setAlternatingRowColors(True)
        self.bad_table.cellDoubleClicked.connect(self._on_bad_activated)
        self.bad_table.setMinimumHeight(160)
        self.bad_table.verticalHeader().setDefaultSectionSize(24)
        self.bad_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.bad_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        bad_card.add(self.bad_table)
        side_layout.addWidget(bad_card, 1)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([760, 430])
        root.addWidget(splitter, 1)

    def _select_defaults(self) -> None:
        available = self.model.metric_run_ids()
        preferred = [
            "503_best_pp",
            "nnUNet_validation_pp",
            "nnUNet_validation_500resenc_pp",
            "nnUNet_validation_500over_pp",
            "nnUNet_validation_501_pp",
            "nnUNet_validation_502_pp",
            "polar_preds_add_t0.7",
        ]
        chosen = [r for r in preferred if r in available][:6]
        for run_id in available:
            if len(chosen) >= 6:
                break
            historical_final = run_id == "503_3d_pp" and "503_best_pp" in available
            if (run_id not in chosen and not run_id.endswith("_grow1")
                    and not historical_final):
                chosen.append(run_id)
        self.runs_list.blockSignals(True)
        for run_id in available:
            item = QListWidgetItem(self.model.run_name(run_id))
            item.setData(Qt.UserRole, run_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if run_id in chosen else Qt.Unchecked)
            item.setToolTip(run_id)
            self.runs_list.addItem(item)
        self.runs_list.blockSignals(False)

    def _set_all(self, checked: bool) -> None:
        self.runs_list.blockSignals(True)
        for i in range(self.runs_list.count()):
            item = self.runs_list.item(i)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.runs_list.blockSignals(False)
        self._replot()

    # ---------------------------------------------------------------- selection
    def selected_runs(self) -> list[str]:
        out = []
        for i in range(self.runs_list.count()):
            item = self.runs_list.item(i)
            if item.checkState() == Qt.Checked:
                out.append(item.data(Qt.UserRole))
        order = {r.run_id: i for i, r in enumerate(self.model.runs)}
        return sorted(out, key=lambda r: order.get(r, 10**6))

    def _on_metric(self, name: str) -> None:
        self.metric_name = name
        self._replot()

    def _on_group(self, name: str) -> None:
        self.group = name
        self._replot()

    def _on_threshold(self, value: float) -> None:
        self.threshold = float(value)
        self._replot()

    def refresh(self) -> None:
        self._replot()

    # ------------------------------------------------------------------- plot
    def _metric_key(self) -> str:
        return "dice" if self.metric_name == "Dice" else "hd95"

    def _score_color(self, value: float | None) -> str:
        if value is None:
            return qt_theme.MUTED
        if self.metric_name == "Dice":
            return qt_theme.score_color(value)
        if value <= 5:
            return qt_theme.GOOD
        if value <= 12:
            return qt_theme.WARN
        return qt_theme.BAD

    def _values(self, run_id: str, label: int) -> list[float]:
        return self.model.metric_values(
            run_id, label, self._metric_key(),
            None if self.group == "All" else self.group)

    @staticmethod
    def _short(name: str, width: int = 20) -> str:
        name = name.replace("nnUNet_validation_", "").replace("nnUNet_", "")
        return name if len(name) <= width else name[: width - 1] + "…"

    def _replot(self) -> None:
        runs = self.selected_runs()
        metric = self._metric_key()
        self.figure.clear()
        if not runs:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Select at least one run", ha="center", va="center",
                    color=qt_theme.MUTED, fontsize=12)
            ax.axis("off")
            self.canvas.draw_idle()
            self._fill_tables(runs)
            return

        colors = [qt_theme.series_color(i) for i in range(len(runs))]
        gs = self.figure.add_gridspec(2, 4, height_ratios=[1.0, 1.0],
                                      hspace=0.42, wspace=0.32,
                                      left=0.07, right=0.985, top=0.90, bottom=0.10)
        ax_bar = self.figure.add_subplot(gs[0, :])
        self._draw_bars(ax_bar, runs, colors)
        for i, label in enumerate(EVAL_LABELS):
            ax = self.figure.add_subplot(gs[1, i])
            self._draw_distribution(ax, runs, colors, label)
        self.figure.suptitle(
            f"{self.metric_name} · {self.group.lower()} cases · "
            f"{len(runs)} run{'s' if len(runs) != 1 else ''}",
            color=qt_theme.TEXT, fontsize=12, fontweight="bold", x=0.07, ha="left",
            y=0.975)
        self.canvas.draw_idle()
        self._fill_tables(runs)

    def _draw_bars(self, ax, runs: list[str], colors: list[str]) -> None:
        metric = self._metric_key()
        n = len(runs)
        width = 0.78 / n
        xs = np.arange(4)
        for i, (run_id, color) in enumerate(zip(runs, colors)):
            means, err_lo, err_hi = [], [], []
            for label in EVAL_LABELS:
                vals = self._values(run_id, label)
                mean, lo, hi = bootstrap_ci(vals)
                means.append(mean)
                err_lo.append(max(0.0, mean - lo) if np.isfinite(mean) else 0.0)
                err_hi.append(max(0.0, hi - mean) if np.isfinite(mean) else 0.0)
            ax.bar(xs + (i - (n - 1) / 2) * width, means, width * 0.92,
                   color=color, alpha=0.9, label=self._short(self.model.run_name(run_id)),
                   yerr=[err_lo, err_hi], capsize=2.0,
                   error_kw={"elinewidth": 0.8, "ecolor": qt_theme.MUTED, "alpha": 0.7})
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL_NAMES[l] for l in EVAL_LABELS])
        ax.set_ylabel(self.metric_name)
        if metric == "dice":
            ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(ncol=min(4, n), fontsize=7, loc="upper center",
                  bbox_to_anchor=(0.5, 1.16), handlelength=1.0, columnspacing=1.2)
        ax.set_title("Mean by class (95% bootstrap CI)", loc="left",
                     color=qt_theme.MUTED, fontsize=9, pad=6)

    def _draw_distribution(self, ax, runs: list[str], colors: list[str],
                           label: int) -> None:
        data = [self._values(run_id, label) for run_id in runs]
        positions = np.arange(len(runs))
        present = [(p, vals, color) for p, vals, color in zip(positions, data, colors)
                   if vals]
        if present:
            parts = ax.violinplot([v for _, v, _ in present],
                                  positions=[p for p, _, _ in present],
                                  widths=0.72, showmeans=False, showextrema=False)
            for body, (_, _, color) in zip(parts["bodies"], present):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.22)
                body.set_linewidth(0.8)
            rng = np.random.default_rng(0)
            for p, vals, color in present:
                groups = self._case_groups(runs[p], label)
                for value, group in zip(vals, groups):
                    ax.scatter(p + rng.uniform(-0.12, 0.12), value, s=9,
                               color=GROUP_COLORS.get(group, qt_theme.MUTED),
                               alpha=0.75, linewidths=0)
            means = [np.mean(v) for _, v, _ in present]
            ax.scatter([p for p, _, _ in present], means, marker="_", s=180,
                       color=qt_theme.TEXT, linewidths=1.4, zorder=5)
        ax.set_title(LABEL_NAMES[label], color=qt_theme.TEXT, fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels([self._short(self.model.run_name(r), 10) for r in runs],
                           rotation=45, ha="right", fontsize=6.5)
        ax.tick_params(axis="y", labelsize=7)
        if self.metric_name == "Dice":
            ax.set_ylim(0, 1.02)
        ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.2)
        ax.set_axisbelow(True)

    def _case_groups(self, run_id: str, label: int) -> list[str]:
        table = self.model.run_table(run_id)
        if table is None:
            return []
        metric = self._metric_key()
        defined = "dice_defined" if metric == "dice" else "hd95_defined"
        groups = []
        for row in table.rows:
            if row.get("class_id") != label:
                continue
            if self.group != "All" and row.get("group") != self.group:
                continue
            value = row.get(metric)
            if not row.get(defined) or value is None or not np.isfinite(value):
                continue
            groups.append(row.get("group") or self.model.group_of(row.get("case_id", "")))
        return groups

    # ----------------------------------------------------------------- tables
    def _fill_tables(self, runs: list[str]) -> None:
        self._fill_scorecard(runs)
        self._fill_bad_cases(runs)

    def _fill_scorecard(self, runs: list[str]) -> None:
        self.score_table.setRowCount(len(runs))
        for i, run_id in enumerate(runs):
            name = QTableWidgetItem(self._short(self.model.run_name(run_id), 24))
            name.setToolTip(run_id)
            self.score_table.setItem(i, 0, name)
            class_means = []
            for j, label in enumerate(EVAL_LABELS, start=1):
                vals = self._values(run_id, label)
                if not vals:
                    item = QTableWidgetItem("—")
                    item.setForeground(QBrush(QColor(qt_theme.MUTED)))
                    self.score_table.setItem(i, j, item)
                    continue
                mean = float(np.mean(vals))
                class_means.append(mean)
                item = QTableWidgetItem(f"{mean:.2f}")
                item.setForeground(QBrush(QColor(self._score_color(mean))))
                item.setToolTip(f"{mean:.3f} ± {np.std(vals):.3f} · n={len(vals)} cases")
                self.score_table.setItem(i, j, item)
            overall = float(np.mean(class_means)) if class_means else None
            item = QTableWidgetItem(
                "—" if overall is None else f"{overall:.3f}")
            if overall is not None:
                item.setForeground(QBrush(QColor(self._score_color(overall))))
            item.setToolTip("Unweighted mean of the four class means")
            self.score_table.setItem(i, 5, item)
        self.score_table.resizeColumnsToContents()

    def _fill_bad_cases(self, runs: list[str]) -> None:
        if not runs:
            self.bad_table.setRowCount(0)
            return
        rows = self.model.bad_cases(
            runs[0], self.threshold,
            None if self.group == "All" else self.group)
        self.bad_table.setRowCount(len(rows))
        for i, entry in enumerate(rows):
            case_item = QTableWidgetItem(entry["case_id"].replace("Case_", ""))
            case_item.setData(Qt.UserRole, entry["case_id"])
            self.bad_table.setItem(i, 0, case_item)
            group = entry.get("group") or self.model.group_of(entry["case_id"])
            group_item = QTableWidgetItem("Path" if group == "Pathological" else group)
            group_item.setToolTip(group)
            self.bad_table.setItem(i, 1, group_item)
            fold = entry.get("fold")
            self.bad_table.setItem(i, 2, QTableWidgetItem(
                "—" if fold is None else str(fold)))
            for j, label in enumerate(EVAL_LABELS, start=3):
                value = entry["dice"].get(label)
                item = QTableWidgetItem(
                    "—" if value is None else f"{value:.2f}")
                if value is not None:
                    item.setForeground(QBrush(QColor(qt_theme.score_color(value))))
                self.bad_table.setItem(i, j, item)
        self.bad_table.resizeColumnsToContents()
        self.hint.setText(
            f"{len(rows)} case(s) below {self.threshold:.2f} on the first selected run"
            f" · double-click to open")

    def _review_failures(self) -> None:
        runs = self.selected_runs()
        if not runs:
            self.status.emit("no run selected")
            return
        cases = [e["case_id"] for e in self.model.bad_cases(
            runs[0], self.threshold,
            None if self.group == "All" else self.group)]
        if not cases:
            self.status.emit(f"no cases below {self.threshold:.2f}")
            return
        self.model.set_queue(cases)
        self.queue_requested.emit(cases)

    def _on_bad_activated(self, row: int, _column: int) -> None:
        item = self.bad_table.item(row, 0)
        if item is not None and item.data(Qt.UserRole):
            self.case_requested.emit(item.data(Qt.UserRole))

    # ----------------------------------------------------------------- export
    def export_png(self, path: str | None = None) -> str:
        out = Path(path) if path else repo_path("var/results/viewer_metrics.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(str(out), dpi=200, facecolor=DARK.figure_face,
                            bbox_inches="tight")
        self.status.emit(f"Saved {out}")
        return str(out)
