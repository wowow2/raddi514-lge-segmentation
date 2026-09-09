"""Case-browsing panel: 3-pane layout (image + filmstrip + HUD).

Owns all browse state (slice, mask, per-class set, window/level, filter) and
exposes action methods. No key handling, no analytics — the viewer root maps
keys to these actions and re-renders.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from src.datautils.emidec_loader import (
    EMIDECCase,
    LABEL_CAVITY,
    LABEL_MVO,
    LABEL_MYOCARDIUM,
    LABEL_SCAR,
)
from .keys import help_lines
from .theme import Theme

FILTERS = ("all", "scar", "mvo", "normal")
THUMB = 64  # filmstrip thumbnail box (px)


def _matches(case: EMIDECCase, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "scar":
        return case.has_scar
    if mode == "mvo":
        return case.has_mvo
    return case.group == "Normal"


class CaseBrowser:
    def __init__(self, fig, cases: list, theme: Theme, on_change, flash):
        """Args:
            fig: parent figure (cleared and laid out by this panel).
            cases: full EMIDECCase list.
            theme: Theme dataclass.
            on_change: callback invoked after any state change (viewer re-renders).
            flash: callback for transient title messages.
        """
        self.fig = fig
        self.theme = theme
        self.all_cases = list(cases)
        if not self.all_cases:
            raise ValueError("CaseBrowser needs at least one case")
        self.on_change = on_change
        self.flash = flash

        self.filter_mode = "all"
        self.visible: list[EMIDECCase] = list(cases)
        self.pos = 0
        self.slice_idx = 0
        self.show_mask = True
        self.visible_labels = {LABEL_CAVITY, LABEL_MYOCARDIUM, LABEL_SCAR, LABEL_MVO}
        self.wl, self.ww = 0.5, 1.0  # window level/width on the 0-1 display image
        self._strip_cache: tuple | None = None  # (case_id, montage, scar_flags)

        self._build_artists()
        self._connect_mouse()

    def rebuild(self):
        """Recreate axes/artists (e.g. returning from dashboard, which clears
        the figure). Mouse connections persist — connected once in __init__."""
        self._build_artists()

    # ------------------------------------------------------------------ layout
    def _build_artists(self):
        t = self.theme
        self.fig.clear()
        gs = self.fig.add_gridspec(
            2, 2, width_ratios=[1.5, 1.0], height_ratios=[4.0, 1.0],
            wspace=0.12, hspace=0.25,
        )
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_strip = self.fig.add_subplot(gs[1, :])
        for ax in (self.ax_img, self.ax_info, self.ax_strip):
            ax.set_facecolor(t.axes_face)
        self.ax_img.axis("off")
        self.ax_info.axis("off")
        self.ax_strip.axis("off")
        self.ax_strip.set_title("Slices (scroll / click)", color=t.dim_text,
                                fontsize=8, pad=2, loc="left")

        case = self.current_case
        img, gt = self._arrays(case)
        self.im_artist = self.ax_img.imshow(
            img[:, :, 0].T, cmap="gray", origin="lower",
            aspect="equal", vmin=0, vmax=1,
        )
        self.mask_artist = self.ax_img.imshow(
            self._masked(gt[:, :, 0].T) if gt is not None else img[:, :, 0].T * 0,
            cmap=t.overlay_cmap, origin="lower", aspect="equal",
            vmin=0, vmax=4, interpolation="none",
        )
        legend_elements = [
            Patch(facecolor=color, edgecolor="none", label=f"{i + 1}: {name}")
            for i, (name, color) in enumerate(t.classes)
        ]
        self.ax_img.legend(
            handles=legend_elements, loc="lower center", ncol=4, fontsize=8,
            frameon=True, facecolor=t.panel_face, edgecolor=t.panel_edge,
            labelcolor=t.text,
        )
        # Scale bar artists (updated per render).
        (self._scale_line,) = self.ax_img.plot([], [], color=t.text, lw=2)
        self._scale_text = self.ax_img.text(
            0, 0, "", color=t.text, fontsize=8, ha="right", va="bottom")
        # Filmstrip artists.
        self.strip_artist = self.ax_strip.imshow(
            np.zeros((THUMB, THUMB)), origin="upper", aspect="equal")
        self.strip_box = Rectangle(
            (0, 0), THUMB, THUMB, fill=False, edgecolor=t.strip_current_edge,
            lw=2)
        self.ax_strip.add_patch(self.strip_box)

    def _connect_mouse(self):
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_scroll(self, event):
        if event.inaxes is self.ax_img:
            self.slice_idx += 1 if event.button == "up" else -1
            _, gt = self._arrays(self.current_case)
            self.slice_idx %= (gt.shape[2] if gt is not None
                               else self.current_case.load()[0].shape[2])
            self.on_change()

    def _on_click(self, event):
        if event.inaxes is self.ax_strip and event.xdata is not None:
            self.slice_idx = max(
                0, min(int(event.xdata // THUMB),
                       self._arrays(self.current_case)[0].shape[2] - 1))
            self.on_change()

    # ------------------------------------------------------------------ state
    @property
    def current_case(self) -> EMIDECCase:
        return self.visible[self.pos]

    def _arrays(self, case: EMIDECCase):
        img = case.display_image
        _, gt = case.load()
        return img, gt

    def _masked(self, gt_slice: np.ndarray) -> np.ndarray:
        vis = sorted(self.visible_labels)
        return np.where(np.isin(gt_slice, vis), gt_slice, 0)

    # ----------------------------------------------------------------- actions
    def slice_step(self, delta: int):
        n = self._arrays(self.current_case)[0].shape[2]
        self.slice_idx = (self.slice_idx + delta) % n

    def case_step(self, delta: int):
        self.pos = (self.pos + delta) % len(self.visible)
        self.slice_idx = 0

    def mask_toggle(self):
        self.show_mask = not self.show_mask

    def class_toggle(self, label: int):
        if label in self.visible_labels:
            self.visible_labels.discard(label)
        else:
            self.visible_labels.add(label)

    def pathology_next(self):
        """Jump to the next slice containing scar/MVO (wraps)."""
        _, gt = self._arrays(self.current_case)
        if gt is None:
            self.flash("No ground truth (blind case)")
            return
        n = gt.shape[2]
        hits = [s for s in range(n)
                if ((gt[:, :, s] == LABEL_SCAR).any()
                    or (gt[:, :, s] == LABEL_MVO).any())]
        if not hits:
            self.flash(f"No scar/MVO slices in {self.current_case.case_id}")
            return
        later = [s for s in hits if s > self.slice_idx]
        self.slice_idx = later[0] if later else hits[0]

    def filter_cycle(self):
        self.filter_mode = FILTERS[(FILTERS.index(self.filter_mode) + 1)
                                   % len(FILTERS)]
        self.visible = [c for c in self.all_cases
                        if _matches(c, self.filter_mode)]
        if not self.visible:  # e.g. test split has no normals
            self.flash(f"No '{self.filter_mode}' cases — filter reset")
            self.filter_mode = "all"
            self.visible = list(self.all_cases)
        self.pos = 0
        self.slice_idx = 0

    def jump_to_global_index(self, idx: int):
        """Position on all_cases[idx]; resets any active filter."""
        if self.filter_mode != "all":
            self.filter_mode = "all"
            self.visible = list(self.all_cases)
        case_id = self.all_cases[idx].case_id
        self.pos = next(i for i, c in enumerate(self.visible)
                        if c.case_id == case_id)
        self.slice_idx = 0

    def window_width(self, delta: int):
        self.ww = float(np.clip(self.ww * (1.25 ** delta), 0.05, 2.0))

    def window_level(self, delta: int):
        self.wl = float(np.clip(self.wl + 0.05 * delta, 0.0, 1.0))

    def export_png(self, path: str | None = None) -> str:
        if path is None:
            os.makedirs("results", exist_ok=True)
            path = (f"results/viewer_{self.current_case.case_id}"
                    f"_s{self.slice_idx + 1}.png")
        self.fig.savefig(path, facecolor=self.theme.figure_face, dpi=150)
        self.flash(f"Saved {path}")
        return path

    # ----------------------------------------------------------------- render
    def _render_scale_bar(self):
        case = self.current_case
        px_per_mm = 1.0 / case.zooms[0]
        for target_mm in (50, 20, 10):
            if target_mm * px_per_mm <= case.shape[0] * 0.4:
                length_mm = target_mm
                break
        else:
            length_mm = 10
        length_px = length_mm * px_per_mm
        x1, y1 = case.shape[1], case.shape[0]
        x0 = x1 - 0.03 * x1 - length_px
        y = 0.05 * y1
        self._scale_line.set_data([x0, x0 + length_px], [y, y])
        self._scale_text.set_text(f"{length_mm} mm")
        self._scale_text.set_position((x0 + length_px, y * 1.6))

    def _render_strip(self, img: np.ndarray, gt: np.ndarray | None):
        t = self.theme
        case = self.current_case
        if self._strip_cache is None or self._strip_cache[0] != case.case_id:
            z = img.shape[2]
            stride = max(1, int(np.ceil(max(img.shape[:2]) / THUMB)))
            thumbs = []
            scar_flags = []
            for s in range(z):
                th = img[::stride, ::stride, s]
                canvas = np.zeros((THUMB, THUMB))
                h, w = th.shape
                canvas[:h, :w] = th
                thumbs.append(canvas)
                scar_flags.append(
                    gt is not None and (
                        (gt[:, :, s] == LABEL_SCAR).any()
                        or (gt[:, :, s] == LABEL_MVO).any()))
            rgb = np.stack(thumbs, axis=1).reshape(THUMB, THUMB * z, 1)
            rgb = np.repeat(rgb, 3, axis=2)
            for s, hit in enumerate(scar_flags):
                if hit:
                    rgb[:3, s * THUMB:(s + 1) * THUMB, 0] = 1.0
                    rgb[:3, s * THUMB:(s + 1) * THUMB, 1:] = 0.15
            self._strip_cache = (case.case_id, rgb, scar_flags)
        _, montage, _ = self._strip_cache
        self.strip_artist.set_data(montage)
        self.strip_artist.set_extent((0, montage.shape[1], THUMB, 0))
        self.ax_strip.set_xlim(0, montage.shape[1])
        self.ax_strip.set_ylim(THUMB, 0)
        self.strip_box.set_xy((self.slice_idx * THUMB, 0))

    def _hud(self, case: EMIDECCase) -> str:
        meta = case.clinical_meta
        vox = case.voxel_counts
        filt = "" if self.filter_mode == "all" else f" [{self.filter_mode}]"
        lines = [
            f"{case.case_id}  ({case.group})",
            f"Case {self.pos + 1}/{len(self.visible)}{filt}",
            "",
            "PATIENT",
            f"  Age / Sex: {meta.get('Age', 'N/A')} yrs | {meta.get('Sex', 'N/A')}",
            f"  Overweight: {meta.get('Overweight', 'N/A')} | Tobacco: {meta.get('Tobacco', 'N/A')}",
            f"  Hypertension: {meta.get('Arterial hypertension', 'N/A')} | Diabetes: {meta.get('Diabetes', 'N/A')}",
            "",
            "BIOMARKERS",
            f"  LVEF: {meta.get('FEVG', 'N/A')} % | Troponin: {meta.get('Troponin', 'N/A')}",
            f"  NT-proBNP: {meta.get('NTProBNP', 'N/A')} | Killip: {meta.get('Killip Max', 'N/A')}",
            f"  ECG (ST+): {meta.get('ECG (ST +)', 'N/A')}",
            "",
            "GEOMETRY",
            f"  Matrix: {case.shape[0]} × {case.shape[1]} × {case.shape[2]}",
            f"  Spacing: {case.zooms[0]} × {case.zooms[1]} × {case.zooms[2]} mm",
            f"  Window: L {self.wl:.2f} W {self.ww:.2f}",
            "",
            "PATHOLOGY",
        ]
        if not case.has_gt:
            lines.append("  Ground truth: N/A (blind)")
        else:
            lines += [
                f"  Scar: {'YES' if case.has_scar else 'NO'} ({vox[LABEL_SCAR]:,d})",
                f"  MVO: {'YES' if case.has_mvo else 'NO'} ({vox[LABEL_MVO]:,d})",
                f"  Cavity: {vox[LABEL_CAVITY]:,d} | Myo: {vox[LABEL_MYOCARDIUM]:,d}",
            ]
        lines += ["", "─" * 32, "CONTROLS"] + help_lines()
        return "\n".join(lines)

    def render(self, title: str):
        """Redraw everything. Title string is composed by the viewer root.

        Requires live axes — call rebuild() first if the figure was cleared
        (e.g. by the dashboard).
        """
        t = self.theme
        case = self.current_case
        img, gt = self._arrays(case)

        self.im_artist.set_data(img[:, :, self.slice_idx].T)
        self.im_artist.set_clim(self.wl - self.ww / 2, self.wl + self.ww / 2)
        if self.show_mask and gt is not None and self.visible_labels:
            self.mask_artist.set_data(self._masked(gt[:, :, self.slice_idx].T))
            self.mask_artist.set_visible(True)
        else:
            self.mask_artist.set_visible(False)
        self._render_scale_bar()
        self._render_strip(img, gt)

        self.ax_img.set_title(title, fontsize=t.title_size,
                              fontweight="bold", pad=10, color=t.text)
        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax_info.set_facecolor(t.axes_face)
        self.ax_info.text(
            0.03, 0.99, self._hud(case), transform=self.ax_info.transAxes,
            fontsize=t.hud_size, verticalalignment="top",
            fontfamily=t.hud_font, color=t.text,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=t.panel_face,
                      edgecolor=t.panel_edge, alpha=0.95),
        )
        self.fig.canvas.draw_idle()
