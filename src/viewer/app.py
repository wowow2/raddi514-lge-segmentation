"""Main window: tabs, toolbar, search, and the minimal key map."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QLineEdit,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QTextBrowser,
    QToolBar,
    QWidget,
)

from . import qt_theme
from .compare_tab import CompareTab
from .explore import ExploreTab
from .metrics_tab import MetricsTab
from .model import ViewerModel


class MainWindow(QMainWindow):
    """EMIDEC explorer: Explore, Metrics, Compare."""

    def __init__(self, model: ViewerModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.model = model
        qt_theme.apply_matplotlib_theme()
        self.setWindowTitle("EMIDEC Explorer · LGE cardiac MRI")
        self.resize(1460, 920)
        self.setMinimumSize(1120, 720)

        self.tabs = QTabWidget()
        self.explore = ExploreTab(model)
        self.metrics = MetricsTab(model)
        self.compare = CompareTab(model)
        self.tabs.addTab(self.explore, "Explore")
        self.tabs.addTab(self.metrics, "Metrics")
        self.tabs.addTab(self.compare, "Compare")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        self.explore.status.connect(self._show_status)
        self.metrics.status.connect(self._show_status)
        self.metrics.queue_requested.connect(self._open_queue)
        self.compare.status.connect(self._show_status)
        self.metrics.case_requested.connect(self._open_case)

        self._build_toolbar()
        self.context = QLabel()
        self.context.setObjectName("Muted")
        self.statusBar().addPermanentWidget(self.context)
        self.explore.canvas.setFocus()
        self._show_status(self.explore.title_text())

    # ---------------------------------------------------------------- toolbar
    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        self.addToolBar(bar)

        brand = QLabel("EMIDEC")
        brand.setStyleSheet(
            f"font-weight:700; letter-spacing:0.12em; color:{qt_theme.TEXT};"
            f"padding-right:10px;")
        bar.addWidget(brand)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search case number…")
        self.search.setFixedWidth(190)
        self.search.returnPressed.connect(self._on_search)
        bar.addWidget(self.search)

        go = QAction("Go", self)
        go.triggered.connect(self._on_search)
        bar.addAction(go)

        find = QAction(self)
        find.setShortcut(QKeySequence("Ctrl+F"))
        find.triggered.connect(self._focus_search)
        self.addAction(find)

        bar.addSeparator()
        prev_action = QAction("‹ Prev", self)
        prev_action.triggered.connect(lambda: self._step_case(-1))
        bar.addAction(prev_action)
        next_action = QAction("Next ›", self)
        next_action.triggered.connect(lambda: self._step_case(1))
        bar.addAction(next_action)

        bar.addSeparator()
        export = QAction("Export PNG", self)
        export.setShortcut(QKeySequence("Ctrl+S"))
        export.triggered.connect(self._export_current)
        bar.addAction(export)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bar.addWidget(spacer)

        help_action = QAction("Help", self)
        help_action.triggered.connect(self._show_help)
        bar.addAction(help_action)

    # -------------------------------------------------------------- navigation
    def _current_view(self):
        widget = self.tabs.currentWidget()
        if hasattr(widget, "slice_step"):
            return widget
        return None

    def _step_case(self, delta: int) -> None:
        view = self.tabs.currentWidget()
        if hasattr(view, "case_step"):
            view.case_step(delta)
        else:
            self.model.step_case(delta)
            self.explore.refresh(reset_slice=True)
        self._show_status(self._context_text())

    def _on_search(self) -> None:
        text = self.search.text()
        if not text.strip():
            self._focus_search()
            return
        if self.model.find_case(text):
            self.tabs.setCurrentWidget(self.explore)
            self.explore.refresh(reset_slice=True)
            self.compare.refresh(reset_slice=False)
            self._show_status(self.explore.title_text())
            self.search.clear()
            self.explore.canvas.setFocus()
        else:
            self._show_status(f"No case matches '{text}'")

    def _focus_search(self) -> None:
        self.search.setFocus()
        self.search.selectAll()

    def _open_case(self, case_id: str) -> None:
        if self.model.find_case(case_id):
            self.tabs.setCurrentWidget(self.explore)
            self.explore.refresh(reset_slice=True)
            self.compare.refresh(reset_slice=False)
            self._show_status(self.explore.title_text())

    def _open_queue(self, _case_ids: list) -> None:
        ids = self.model.queue_ids()
        current = self.model.current_case
        if current is None or current.case_id not in ids:
            target = self.model.case_by_id(ids[0])
            if target is not None:
                self.model.set_case_view("All cases", "Case ID")
                for i, case in enumerate(self.model.cases):
                    if case.case_id == ids[0]:
                        self.model.set_case_index(i)
                        break
        self.tabs.setCurrentWidget(self.explore)
        self.explore.refresh(reset_slice=True)
        self.compare.refresh(reset_slice=False)
        self._show_status(f"review queue {len(ids)} case(s) loaded")

    def _on_tab_changed(self, _index: int) -> None:
        widget = self.tabs.currentWidget()
        if widget is self.metrics:
            self.metrics.refresh()
        elif widget is self.compare:
            self.compare.refresh(reset_slice=False)
        elif widget is self.explore:
            self.explore.refresh(reset_slice=False)
        widget.setFocus()
        self._show_status(self._context_text())

    # ------------------------------------------------------------------ status
    def _show_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 6000)
        self.context.setText(self._context_text())

    def _context_text(self) -> str:
        case = self.model.current_case
        return (f"{case.case_id} · {case.group} · "
                f"slice {self.model.slice_idx + 1}/{self.model.slice_count()}")

    def _export_current(self) -> None:
        widget = self.tabs.currentWidget()
        if hasattr(widget, "export_png"):
            path = widget.export_png()
            self._show_status(f"Saved {path}")
        else:
            self._show_status("Nothing to export on this tab")

    # ------------------------------------------------------------------- help
    def _show_help(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Using the explorer")
        box.setTextFormat(Qt.RichText)
        box.setText(
            "<h3 style='margin:0 0 8px'>Mouse first</h3>"
            "<p style='margin:0 0 10px'>Every control is on screen. The class "
            "checkboxes choose what is drawn; <b>Focus</b> selects the class used "
            "for the probability heatmap and the GT→prediction breakdown.</p>"
            "<h3 style='margin:0 0 6px'>Explore</h3>"
            "<ul style='margin-top:0'>"
            "<li><b>Overlay</b>: GT, Prediction, Both (GT fill + prediction "
            "contour), Error (TP/FP/FN), or a stage-1 probability heatmap.</li>"
            "<li><b>Style</b>: Mask, Contour, or Both.</li>"
            "<li><b>Layout</b>: Single slice or a whole-volume montage. Click a "
            "montage tile to select that slice.</li>"
            "<li><b>View</b>: LGE or display-only derived channels "
            "(local contrast, LoG, local standard deviation, p5–p95 window).</li>"
            "<li>The right panel shows patient metadata, ground-truth volumes, "
            "per-class Dice/HD95/recall/precision, and the GT→prediction "
            "confusion for the focus class.</li>"
            "<li><b>Gate preview</b>: dashed outlines mark prediction "
            "components the challenge gates would zero (scar &lt; 50 vox, "
            "MVO &lt; 10, MVO-without-scar), and the panel reports the "
            "post-gate Dice.</li>"
            "<li><b>Review queue</b>: Metrics → <i>Review failures</i> loads "
            "every case below the threshold; ‹/Next steps through just those "
            "cases until you exit the queue.</li>"
            "</ul>"
            "<h3 style='margin:0 0 6px'>Metrics</h3>"
            "<ul style='margin-top:0'>"
            "<li>Check any set of runs; plots and the scorecard update.</li>"
            "<li>Bars show mean with a 95% bootstrap CI, dots are cases coloured "
            "by group.</li>"
            "<li>Double-click a failure-triage row to open that case.</li>"
            "</ul>"
            "<h3 style='margin:0 0 6px'>Compare</h3>"
            "<p style='margin:0 0 10px'>Two linked panes for the same case and "
            "slice, each with its own run. Works with single slices or montages.</p>"
            "<h3 style='margin:0 0 6px'>Shortcuts</h3>"
            "<table cellspacing='0' cellpadding='3'>"
            "<tr><td><b>← / →</b></td><td>previous / next slice</td></tr>"
            "<tr><td><b>↑ / ↓</b></td><td>previous / next case</td></tr>"
            "<tr><td><b>Space</b></td><td>cycle mask → contour → both</td></tr>"
            "<tr><td><b>Ctrl+F</b></td><td>search box</td></tr>"
            "<tr><td><b>Ctrl+S</b></td><td>export the current tab</td></tr>"
            "<tr><td><b>Esc</b></td><td>leave the search box</td></tr>"
            "</table>")
        box.exec_()

    # ------------------------------------------------------------------ input
    def keyPressEvent(self, event) -> None:
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QComboBox, QSlider, QTextBrowser)):
            if event.key() == Qt.Key_Escape:
                focus.clearFocus()
                self.explore.canvas.setFocus()
                return
            super().keyPressEvent(event)
            return
        view = self.tabs.currentWidget()
        if event.key() in (Qt.Key_Left, Qt.Key_Right) and hasattr(view, "slice_step"):
            view.slice_step(-1 if event.key() == Qt.Key_Left else 1)
            self._show_status(self._context_text())
        elif event.key() in (Qt.Key_Up, Qt.Key_Down) and hasattr(view, "case_step"):
            view.case_step(-1 if event.key() == Qt.Key_Up else 1)
            self._show_status(self._context_text())
        elif event.key() == Qt.Key_Space and hasattr(view, "cycle_style"):
            view.cycle_style()
        else:
            super().keyPressEvent(event)
