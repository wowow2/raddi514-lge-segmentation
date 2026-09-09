"""DatasetViewer: orchestration root. Owns the figure, browse/dashboard mode,
global key dispatch, and case-number search. Rendering lives in CaseBrowser
and AnalyticsDashboard; analytics compute lives in DatasetAnalytics.
"""

import os
import re

import matplotlib.pyplot as plt

from src.datautils.analytics import DatasetAnalytics
from .browser import CaseBrowser
from .dashboard import AnalyticsDashboard
from .keys import BROWSER_KEYS, VIEWER_KEYS, find_action
from .theme import DARK


class DatasetViewer:
    def __init__(self, cases, analytics_cache: str | None = None):
        self.cases = list(cases)
        if not self.cases:
            raise ValueError("DatasetViewer needs at least one case")
        self.theme = DARK
        self.analytics_cache = (analytics_cache
                                or DatasetAnalytics.cache_path_for(self.cases))
        self.mode = "browse"  # or "dashboard"
        self._search: str | None = None
        self._flash_msg: str | None = None
        self._stats: dict | None = None

        self.fig = plt.figure(figsize=(13, 8))
        self.fig.patch.set_facecolor(self.theme.figure_face)
        self.fig.canvas.manager.set_window_title(
            "EMIDEC Explorer - Imaging & Clinical Metadata")

        self.browser = CaseBrowser(
            self.fig, self.cases, self.theme,
            on_change=self.render, flash=self.flash)
        self.dashboard = AnalyticsDashboard(self.fig, self.theme)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.render()

    # ------------------------------------------------------------- messaging
    def flash(self, msg: str):
        self._flash_msg = msg

    # ---------------------------------------------------------------- search
    @staticmethod
    def _case_number(case_id: str) -> str:
        m = re.search(r"(\d+)$", case_id)
        return m.group(1) if m else ""

    def _search_matches(self, buf: str) -> list[int]:
        ids = [self._case_number(c.case_id) for c in self.browser.all_cases]
        if not buf:
            return list(range(len(ids)))
        b = buf.lstrip("0")
        if not b:
            return [i for i, num in enumerate(ids) if num.startswith("0")]
        return [i for i, num in enumerate(ids)
                if num.lstrip("0").startswith(b)]

    def _jump_to_match(self):
        matches = self._search_matches(self._search or "")
        if not matches:
            self.flash(f"No match for '{self._search}'")
            self._search = None
            return
        ordered = sorted(matches,
                         key=lambda i: (i - self.browser.pos) % len(self.browser.all_cases))
        self._flash_msg = None
        self._search = None
        self.browser.jump_to_global_index(ordered[0])

    # ------------------------------------------------------------- dashboard
    def _toggle_dashboard(self):
        if self.mode == "browse":
            self.mode = "dashboard"
            if self._stats is None:
                try:
                    self._stats = DatasetAnalytics.load(self.analytics_cache)
                    print(f"Loaded analytics cache: {self.analytics_cache}")
                except (FileNotFoundError, ValueError, KeyError):
                    print("Computing analytics (first run, cached after)…")
                    analyzer = DatasetAnalytics(self.browser.all_cases,
                                                cache_path=self.analytics_cache)
                    self._stats = analyzer.compute()
                    analyzer.save()
            self.dashboard.render(self._stats)
        else:
            self.mode = "browse"
            self.browser.rebuild()
            self.render()

    # ---------------------------------------------------------------- render
    def _title(self) -> str:
        if self._search is not None:
            n = len(self._search_matches(self._search))
            return (f"Go to case: {self._search}▏ "
                    f"({n} match{'es' if n != 1 else ''})")
        if self._flash_msg is not None:
            msg = self._flash_msg
            self._flash_msg = None
            return msg
        case = self.browser.current_case
        n_slices = case.load()[0].shape[2]
        filt = ("" if self.browser.filter_mode == "all"
                else f" [{self.browser.filter_mode}]")
        return (f"{case.case_id} ({case.group}) — "
                f"Slice {self.browser.slice_idx + 1}/{n_slices} — "
                f"Case {self.browser.pos + 1}/{len(self.browser.visible)}{filt}")

    def render(self):
        if self.mode == "dashboard":
            return  # dashboard is static until toggled off
        self.browser.render(self._title())

    # ----------------------------------------------------------------- input
    def _on_search_key(self, key: str):
        if key == "enter":
            self._jump_to_match()
        elif key == "escape":
            self._search = None
        elif key == "backspace":
            self._search = self._search[:-1]
        elif key in "0123456789":
            self._search += key
        else:
            return
        self.render()

    def _on_browser_key(self, key: str):
        action = find_action(key, BROWSER_KEYS)
        if action is None:
            return
        b = self.browser
        if action == "slice_next":
            b.slice_step(1)
        elif action == "slice_prev":
            b.slice_step(-1)
        elif action == "case_next":
            b.case_step(1)
        elif action == "case_prev":
            b.case_step(-1)
        elif action == "mask_toggle":
            b.mask_toggle()
        elif action == "class_toggle":
            b.class_toggle(int(key))
        elif action == "pathology_next":
            b.pathology_next()
        elif action == "filter_cycle":
            b.filter_cycle()
        elif action == "window_width":
            b.window_width(1 if key == "equal" else -1)
        elif action == "window_level":
            b.window_level(1 if key == "bracketright" else -1)
        elif action == "export_png":
            b.export_png()
        else:
            return
        self.render()

    def on_key(self, event):
        if self._search is not None:
            self._on_search_key(event.key)
            return
        if self.mode == "dashboard":
            if event.key in ("q", "Q"):
                plt.close(self.fig)
            else:  # any other key (incl. Esc, A) returns to browse
                self._toggle_dashboard()
            return
        action = find_action(event.key, VIEWER_KEYS)
        if action == "search_start":
            self._search = ""
            self.render()
        elif action == "dashboard_toggle":
            self._toggle_dashboard()
        elif action == "quit":
            plt.close(self.fig)
        else:
            self._on_browser_key(event.key)

    def show(self):
        plt.tight_layout()
        plt.show()
