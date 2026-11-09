"""Percentile-window plot of per-case metric distributions."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import LABEL_NAMES
from src.results import ResultsTable


class PercentileWindowPlot:
    """For each method: 0–100% window, 25–75% box and median marker."""

    def __init__(self, tables: dict[str, ResultsTable],
                 labels: tuple[int, ...] = (1, 2, 3, 4),
                 metric: str = "dice", gt_present: bool = False):
        self.tables = tables
        self.labels = labels
        self.metric = metric
        self.gt_present = gt_present
        self._data: dict[str, dict[int, list[float]]] | None = None

    def collect(self):
        data = {name: {lbl: [] for lbl in self.labels}
                for name in self.tables}
        defined = f"{self.metric}_defined"
        for name, table in self.tables.items():
            for row in table.rows:
                lbl = row.get("class_id")
                if lbl not in data[name]:
                    continue
                if not row.get(defined):
                    continue
                if self.gt_present and not row.get("gt_present"):
                    continue
                value = row.get(self.metric)
                if value is None or not np.isfinite(value):
                    continue
                data[name][lbl].append(float(value))
        self._data = data
        return data

    def save(self, path: str, title: str | None = None):
        if self._data is None:
            self.collect()
        names = list(self.tables)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 1)))
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
        axes = np.ravel(axes)
        metric_label = "Dice" if self.metric == "dice" else "HD95 (mm)"
        ylim = (-0.05, 1.05) if self.metric == "dice" else None
        for ax, lbl in zip(axes, self.labels):
            plotted = False
            for xi, name in enumerate(names):
                vals = self._data[name][lbl]
                if not vals:
                    continue
                p0, p25, p50, p75, p100 = np.percentile(
                    vals, [0, 25, 50, 75, 100])
                color = colors[xi]
                ax.vlines(xi, p0, p100, color=color, lw=1.5, alpha=0.55)
                ax.vlines(xi, p25, p75, color=color, lw=8, alpha=0.85)
                ax.plot(xi, p50, "o", color="white", markeredgecolor=color,
                        markeredgewidth=1.5, markersize=6, zorder=3)
                ax.text(xi + 0.10, p50, f"{p50:.2f}", color=color,
                        fontsize=7, va="center")
                plotted = True
            ax.set_title(f"{LABEL_NAMES[lbl]} {metric_label}",
                         color="white", fontweight="bold")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=25, ha="right")
            ax.set_ylabel(metric_label)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.2)
            if not plotted:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", color="white")
        for ax in axes[len(self.labels):]:
            ax.axis("off")
        fig.patch.set_facecolor("#1e1e1e")
        for ax in axes:
            ax.set_facecolor("#252525")
            ax.tick_params(colors="white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_color("#555555")
        handles = [
            plt.Line2D([], [], color="white", lw=1.5, alpha=0.55,
                       label="0–100%"),
            plt.Line2D([], [], color="white", lw=8, alpha=0.85,
                       label="25–75%"),
            plt.Line2D([], [], marker="o", color="white", linestyle="",
                       label="median"),
        ]
        fig.legend(handles=handles, loc="upper right", facecolor="#2e2e2e",
                   edgecolor="#555555", labelcolor="white")
        fig.suptitle(title or f"Per-case {metric_label} percentile windows",
                     color="white", fontweight="bold")
        fig.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, facecolor=fig.get_facecolor(), dpi=150)
        print(f"saved -> {path}")
        plt.close(fig)
