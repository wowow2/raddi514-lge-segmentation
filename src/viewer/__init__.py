"""Viewer package: a Qt-free data layer plus the PyQt5 application.

Importing :mod:`src.viewer.model` or :mod:`src.viewer.theme` does not pull in
Qt; the application lives in :mod:`src.viewer.app`.
"""

from .theme import DARK, Theme

__all__ = ["DARK", "Theme"]
