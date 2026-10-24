"""Small reusable Qt widgets for the viewer."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import DARK


class SegmentedControl(QWidget):
    """A compact exclusive button group (e.g. Mask / Contour / Both)."""

    changed = pyqtSignal(str)

    def __init__(self, options: list[str], value: str | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        for option in options:
            button = QPushButton(option)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            if option == value:
                button.setChecked(True)
            self._group.addButton(button)
            self._buttons[option] = button
            layout.addWidget(button)
            button.clicked.connect(lambda _=False, o=option: self.changed.emit(o))
        if value is None and options:
            self._buttons[options[0]].setChecked(True)

    def value(self) -> str:
        for option, button in self._buttons.items():
            if button.isChecked():
                return option
        return ""

    def set_value(self, value: str) -> None:
        if value in self._buttons:
            self._buttons[value].setChecked(True)

    def set_options_enabled(self, enabled: bool) -> None:
        for button in self._buttons.values():
            button.setEnabled(enabled)


class ClassToggles(QWidget):
    """Four class visibility toggles with colour-coded indicators."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self._checks: dict[int, QCheckBox] = {}
        for label, color in enumerate(DARK.legend_colors, start=1):
            check = QCheckBox(DARK.classes[label - 1][0])
            check.setChecked(True)
            check.setCursor(Qt.PointingHandCursor)
            rgb = tuple(round(255 * c) for c in color[:3])
            check.setStyleSheet(
                "QCheckBox::indicator:checked {"
                f"background-color: rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 230);"
                f"border-color: rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 255);"
                "}")
            check.stateChanged.connect(lambda _=0: self.changed.emit())
            self._checks[label] = check
            layout.addWidget(check)

    def visible(self) -> set[int]:
        return {label for label, check in self._checks.items() if check.isChecked()}

    def set_visible(self, labels: set[int]) -> None:
        for label, check in self._checks.items():
            check.blockSignals(True)
            check.setChecked(label in labels)
            check.blockSignals(False)

    def set_enabled_labels(self, enabled: set[int]) -> None:
        for label, check in self._checks.items():
            check.setEnabled(label in enabled)


class Card(QFrame):
    """A titled panel with the shared card styling."""

    def __init__(self, title: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(14, 12, 14, 12)
        self.body.setSpacing(8)
        if title:
            label = QLabel(title.upper())
            label.setObjectName("CardTitle")
            self.body.addWidget(label)

    def add(self, widget: QWidget) -> QWidget:
        self.body.addWidget(widget)
        return widget


class Field(QWidget):
    """Label + widget in a horizontal row (compact control)."""

    def __init__(self, text: str, widget: QWidget, parent: QWidget | None = None,
                 stretch: bool = False):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        label = QLabel(text)
        label.setObjectName("Muted")
        layout.addWidget(label)
        layout.addWidget(widget, 1 if stretch else 0)
        if stretch:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
