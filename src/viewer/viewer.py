"""DatasetViewer: orchestration root. Owns the figure, global key dispatch,
and case-number search. All rendering lives in CaseBrowser.
"""

import re

import matplotlib.pyplot as plt

from .browser import CaseBrowser
from .keys import BROWSER_KEYS, VIEWER_KEYS, find_action
from .theme import DARK


class DatasetViewer:
    def __init__(self, cases, run_names=None, pred_index=None):
        self.cases = list(cases)
        if not self.cases:
            raise ValueError("DatasetViewer needs at least one case")
        self.theme = DARK
        self._search: str | None = None
        self._flash_msg: str | None = None

        self.fig = plt.figure(figsize=(13, 8))
        self.fig.patch.set_facecolor(self.theme.figure_face)
        self.fig.canvas.manager.set_window_title(
            "EMIDEC Explorer - Imaging & Clinical Metadata")

        self.browser = CaseBrowser(
            self.fig, self.cases, self.theme,
            on_change=self.render, flash=self.flash,
            run_names=run_names, pred_index=pred_index)

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
        self.browser.jump_to_index(ordered[0])

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
        title = (f"{case.case_id} ({case.group}) — "
                 f"Slice {self.browser.slice_idx + 1}/{n_slices} — "
                 f"Case {self.browser.pos + 1}/{len(self.browser.all_cases)}")
        if self.browser.overlay != "GT":
            title += f" — [{self.browser.overlay}]"
        return title

    def render(self):
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
        elif action == "pred_cycle":
            b.pred_cycle()
        elif action == "class_toggle":
            b.class_toggle(int(key))
        elif action == "export_png":
            b.export_png()
        else:
            return
        self.render()

    def on_key(self, event):
        if self._search is not None:
            self._on_search_key(event.key)
            return
        action = find_action(event.key, VIEWER_KEYS)
        if action == "search_start":
            self._search = ""
            self.render()
        elif action == "quit":
            plt.close(self.fig)
        else:
            self._on_browser_key(event.key)

    def show(self):
        plt.tight_layout()
        plt.show()
