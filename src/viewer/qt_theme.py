"""Qt styling: one dark, calm palette shared by widgets and matplotlib."""

from __future__ import annotations

from matplotlib import rcParams

from .theme import DARK

PANEL = "#141a23"
PANEL_ALT = "#10151d"
BORDER = "#263041"
HOVER = "#1f2a3a"
ACCENT = "#5bc8ff"
TEXT = "#e7ebf3"
MUTED = "#8d97ab"
GOOD = "#3ddc84"
WARN = "#ffb454"
BAD = "#ff5d5d"

SERIES = (
    "#5bc8ff", "#ff7a59", "#3ddc84", "#c792ea", "#ffd166",
    "#4dd0c1", "#f78fb3", "#9aa5ff", "#e0a458", "#8bd450",
)

CHIP_QSS = """
QLabel#ChipLabel {
    background: rgba(91, 200, 255, 0.10);
    border: 1px solid rgba(91, 200, 255, 0.55);
    border-radius: 10px;
    padding: 3px 12px;
    color: #5bc8ff;
    font-weight: 600;
}
"""


STYLE_SHEET = f"""
QMainWindow, QWidget {{
    background: {DARK.figure_face};
    color: {TEXT};
    font-family: "Inter", "Segoe UI", "DejaVu Sans", sans-serif;
    font-size: 10pt;
}}
QToolBar {{
    background: {PANEL_ALT};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px 10px;
    spacing: 8px;
}}
QToolBar QLabel {{ color: {MUTED}; }}
QStatusBar {{
    background: {PANEL_ALT};
    border-top: 1px solid {BORDER};
    color: {MUTED};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background: {DARK.figure_face};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    padding: 8px 18px;
    margin-right: 4px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QFrame#Card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QLabel#CardTitle {{
    color: {MUTED};
    font-size: 8.5pt;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
QLabel#Muted {{ color: {MUTED}; }}
QComboBox, QLineEdit {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 9px;
    selection-background-color: {ACCENT};
    min-height: 20px;
}}
QComboBox:hover, QLineEdit:hover {{ border-color: #3a4a63; }}
QComboBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    selection-background-color: {HOVER};
    color: {TEXT};
    outline: none;
}}
QPushButton {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 11px;
    color: {TEXT};
}}
QPushButton:hover {{ background: {HOVER}; border-color: #3a4a63; }}
QPushButton:pressed {{ background: #263246; }}
QPushButton:checked {{
    background: #16324a;
    border-color: {ACCENT};
    color: {ACCENT};
}}
QPushButton#Ghost {{
    background: transparent;
    border: 1px solid transparent;
    color: {MUTED};
    padding: 4px 8px;
}}
QPushButton#Ghost:hover {{ color: {TEXT}; background: {HOVER}; }}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{
    width: 13px; height: 13px;
    border-radius: 3px;
    border: 1px solid {BORDER};
    background: {PANEL_ALT};
}}
QCheckBox::indicator:checked {{ border-color: {ACCENT}; }}
QListWidget::indicator, QTableView::indicator {{
    width: 13px; height: 13px;
    border-radius: 3px;
    border: 1px solid {BORDER};
    background: {PANEL_ALT};
}}
QListWidget::indicator:checked, QTableView::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QListWidget::indicator:hover, QTableView::indicator:hover {{
    border-color: {ACCENT};
}}
QSlider::groove:horizontal {{
    height: 3px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {TEXT};
    width: 12px; height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSplitter::handle {{ background: transparent; }}
QTextBrowser {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 10px 12px;
    selection-background-color: #2a3c55;
}}
QListWidget, QTableWidget {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    gridline-color: {BORDER};
    selection-background-color: #1d3550;
    alternate-background-color: {PANEL_ALT};
    outline: none;
}}
QHeaderView::section {{
    background: {PANEL_ALT};
    color: {MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableWidget::item {{ padding: 4px 6px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: #2c394d; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: #3a4a63; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{
    background: #2c394d; border-radius: 5px; min-width: 24px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QToolTip {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 5px;
}}
QDialog {{ background: {DARK.figure_face}; }}
""" + CHIP_QSS


def apply_matplotlib_theme() -> None:
    """Keep embedded figures visually consistent with the Qt chrome."""
    rcParams.update({
        "figure.facecolor": DARK.figure_face,
        "savefig.facecolor": DARK.figure_face,
        "axes.facecolor": DARK.figure_face,
        "axes.edgecolor": DARK.figure_face,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": BORDER,
        "grid.alpha": 0.6,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.facecolor": DARK.figure_face,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })


def score_color(value: float | None) -> str:
    if value is None:
        return MUTED
    if value >= 0.8:
        return GOOD
    if value >= 0.5:
        return WARN
    return BAD


def series_color(index: int) -> str:
    return SERIES[index % len(SERIES)]