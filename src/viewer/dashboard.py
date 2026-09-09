"""Analytics dashboard panel: renders DatasetAnalytics stats as 6 plots.

Pure render — all numbers come from the stats dict. No dataset access.
"""

import numpy as np

from .theme import Theme

CLASS_NAMES = {1: "Cavity", 2: "Myo", 3: "Scar", 4: "MVO"}
MAX_VIOLIN_POINTS = 20000


class AnalyticsDashboard:
    def __init__(self, fig, theme: Theme):
        self.fig = fig
        self.theme = theme

    def _style(self, ax, xlabel: str = "", ylabel: str = ""):
        t = self.theme
        ax.set_facecolor(t.axes_face)
        ax.tick_params(colors=t.dim_text, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(t.panel_edge)
        ax.set_xlabel(xlabel, color=t.dim_text, fontsize=8)
        ax.set_ylabel(ylabel, color=t.dim_text, fontsize=8)
        ax.title.set_color(t.text)
        ax.title.set_fontsize(10)
        ax.grid(True, alpha=0.15, color=t.dim_text)

    def render(self, stats: dict):
        """Draw the 6-panel dashboard from a DatasetAnalytics stats dict."""
        t = self.theme
        self.fig.clear()
        self.fig.suptitle(
            f"EMIDEC Analytics — {stats['n_cases']} cases",
            color=t.text, fontsize=13, fontweight="bold",
        )
        gs = self.fig.add_gridspec(2, 3, hspace=0.45, wspace=0.3,
                                   left=0.06, right=0.96, top=0.88, bottom=0.1)
        axes = [self.fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
        self._voxels(axes[0], stats)
        self._intensities(axes[1], stats)
        self._burden(axes[2], stats)
        self._lvef(axes[3], stats)
        self._cohort(axes[4], stats)
        self._slices(axes[5], stats)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------- panels
    def _voxels(self, ax, stats: dict):
        t = self.theme
        counts = stats["voxel_counts"]
        names = ["BG", "Cavity", "Myo", "Scar", "MVO"]
        vals = [counts[str(i)] for i in range(5)]
        if max(vals) <= 0:
            self._notice(ax, "no ground-truth data")
            ax.set_title("Voxel distribution")
            self._style(ax)
            return
        colors = ["#3a4356"] + list(t.dash_colors[:4])
        ax.bar(names, vals, color=colors, edgecolor=t.panel_edge)
        ax.set_yscale("log")
        total = sum(vals) or 1
        for n, v in zip(names, vals):
            ax.text(n, v, f"{100 * v / total:.2f}%", ha="center", va="bottom",
                    fontsize=7, color=t.dim_text)
        ax.set_title("Voxel distribution")
        self._style(ax, ylabel="voxels (log)")

    def _notice(self, ax, msg: str):
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center",
                va="center", fontsize=9, color=self.theme.dim_text)

    def _intensities(self, ax, stats: dict):
        t = self.theme
        data, labels = [], []
        for lbl in (1, 2, 3, 4):
            vals = np.asarray(stats["intensities"][str(lbl)], dtype=float)
            if vals.size > MAX_VIOLIN_POINTS:
                vals = vals[:: vals.size // MAX_VIOLIN_POINTS]
            if vals.size:
                data.append(vals)
                labels.append(CLASS_NAMES[lbl])
        if not data:
            self._notice(ax, "no ground-truth data")
            ax.set_title("Intensity per label (raw)")
            self._style(ax)
            return
        parts = ax.violinplot(data, showmeans=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(t.dash_colors[i % len(t.dash_colors)])
            pc.set_alpha(0.7)
            pc.set_edgecolor(t.panel_edge)
        for key in ("cmeans", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color(t.text)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_title("Intensity per label (raw)")
        self._style(ax, ylabel="intensity")

    def _burden(self, ax, stats: dict):
        t = self.theme
        burdens = [c["scar_pct_myo"] for c in stats["per_case"] if c["has_scar"]]
        if not burdens:
            self._notice(ax, "no scar cases")
            ax.set_title("Scar burden (% myocardium)")
            self._style(ax)
            return
        ax.hist(burdens, bins=15, color=t.dash_colors[2], edgecolor=t.panel_edge)
        ax.axvline(np.median(burdens), color=t.accent, ls="--", lw=1,
                   label=f"median {np.median(burdens):.1f}%")
        ax.legend(fontsize=7, facecolor=t.panel_face, edgecolor=t.panel_edge,
                  labelcolor=t.text)
        ax.set_title("Scar burden (% myocardium)")
        self._style(ax, xlabel="% of myocardium", ylabel="cases")

    def _lvef(self, ax, stats: dict):
        t = self.theme
        for group, marker, color in (("Pathological", "o", t.dash_colors[2]),
                                     ("Normal", "s", t.dash_colors[1]),
                                     ("Unknown", "^", t.dim_text)):
            xs = [c["scar_pct_myo"] for c in stats["per_case"]
                  if c["group"] == group and c["lvef"] is not None]
            ys = [c["lvef"] for c in stats["per_case"]
                  if c["group"] == group and c["lvef"] is not None]
            if xs:
                ax.scatter(xs, ys, s=14, marker=marker, color=color,
                           alpha=0.8, label=group, edgecolors=t.panel_edge,
                           linewidths=0.5)
        ax.legend(fontsize=7, facecolor=t.panel_face, edgecolor=t.panel_edge,
                  labelcolor=t.text)
        ax.set_title("LVEF vs scar burden")
        self._style(ax, xlabel="scar % of myocardium", ylabel="LVEF (%)")

    def _cohort(self, ax, stats: dict):
        t = self.theme
        cases = stats["per_case"]
        n_norm = sum(1 for c in cases if c["group"] == "Normal")
        n_scar = sum(1 for c in cases if c["has_scar"] and not c["has_mvo"])
        n_mvo = sum(1 for c in cases if c["has_mvo"])
        n_blind = sum(1 for c in cases if not c["has_gt"])
        names, vals, colors = [], [], []
        if n_norm:
            names.append("Normal")
            vals.append(n_norm)
            colors.append(t.dash_colors[1])
        if n_scar:
            names.append("Scar")
            vals.append(n_scar)
            colors.append(t.dash_colors[2])
        if n_mvo:
            names.append("Scar+MVO")
            vals.append(n_mvo)
            colors.append(t.dash_colors[3])
        if n_blind:
            names.append("Blind")
            vals.append(n_blind)
            colors.append(t.dim_text)
        ax.bar(names, vals, color=colors, edgecolor=t.panel_edge)
        for n, v in zip(names, vals):
            ax.text(n, v, str(v), ha="center", va="bottom",
                    fontsize=9, color=t.text)
        ax.set_title("Cohort composition")
        self._style(ax, ylabel="cases")

    def _slices(self, ax, stats: dict):
        t = self.theme
        inv = stats["slice_involvement"]
        hits = np.array(inv["scar_slices"], dtype=float)
        totals = np.array(inv["total_slices"], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.nan_to_num(
                np.where(totals, 100 * hits / np.maximum(totals, 1), np.nan),
                nan=0.0)
        edges = np.array(inv["bin_edges"])
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, rate, width=0.08, color=t.dash_colors[2],
               edgecolor=t.panel_edge)
        shapes = {tuple(c["shape_xy"]) for c in stats["per_case"]}
        spacings = {tuple(c["spacing_xy"]) for c in stats["per_case"]}
        ax.text(0.98, 0.96,
                f"{len(shapes)} shapes · {len(spacings)} spacings",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=t.dim_text)
        ax.set_title("Scar-slice rate base → apex")
        self._style(ax, xlabel="normalized slice position", ylabel="% slices w/ scar")
