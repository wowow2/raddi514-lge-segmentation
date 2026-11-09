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
                 metric: str = "dice"):
        self.tables = tables
        self.labels = labels
        self.metric = metric

    def _values(self, name: str, label: int) -> list[float]:
        table = self.tables[name].filter(class_id=label)
        if label in (3, 4):
            table = table.filter(gt_present=True)
        defined = f"{self.metric}_defined"
        return [float(r[self.metric]) for r in table.rows
                if r.get(defined) and r.get(self.metric) is not None]

    def save(self, path: str):
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        })
        names = list(self.tables)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 1)))
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = np.ravel(axes)
        metric_label = "Dice" if self.metric == "dice" else "HD95 (mm)"
        for ax, label in zip(axes, self.labels):
            for yi, name in enumerate(names):
                values = self._values(name, label)
                if not values:
                    continue
                p0, p25, p50, p75, p100 = np.percentile(
                    values, [0, 25, 50, 75, 100])
                color = colors[yi]
                ax.hlines(yi, p0, p100, color=color, lw=1.0, alpha=0.65)
                ax.hlines(yi, p25, p75, color=color, lw=6, alpha=0.85)
                ax.plot(p50, yi, "o", color="white", markeredgecolor=color,
                        markeredgewidth=1.2, markersize=5, zorder=3)
                ax.text(min(p50 + 0.02, 0.97), yi, f"{p50:.2f}",
                        color=color, fontsize=7, va="center",
                        bbox=dict(facecolor="white", alpha=0.75,
                                  edgecolor="none", pad=0.8))
            title = LABEL_NAMES[label]
            if label in (3, 4):
                title += " (GT-present)"
            ax.set_title(title)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_xlabel(metric_label)
            ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        for ax in axes[len(self.labels):]:
            ax.axis("off")
        handles = [
            plt.Line2D([], [], color="black", lw=1.0, alpha=0.65,
                       label="0–100%"),
            plt.Line2D([], [], color="black", lw=6, alpha=0.85,
                       label="25–75%"),
            plt.Line2D([], [], marker="o", color="white",
                       markeredgecolor="black", linestyle="",
                       label="median"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   frameon=False, fontsize=8)
        fig.suptitle(f"Per-case {metric_label} percentile windows")
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved -> {path}")
        plt.close(fig)
