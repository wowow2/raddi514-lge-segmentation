"""Grouped per-class mean Dice and HD95 bars for several runs."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import LABEL_NAMES
from src.results import ResultsTable


class SummaryBarsPlot:
    def __init__(self, tables: dict[str, ResultsTable],
                 labels: tuple[int, ...] = (1, 2, 3, 4)):
        self.tables = tables
        self.labels = labels

    def _stats(self, name: str, label: int, metric: str):
        table = self.tables[name].filter(class_id=label)
        return table.overall(metric)

    def save(self, path: str):
        names = list(self.tables)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 1)))
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        for ax, metric, ylabel in (
                (axes[0], "dice", "Dice"),
                (axes[1], "hd95", "HD95 (mm)")):
            x = np.arange(len(self.labels))
            width = 0.8 / max(len(names), 1)
            for i, name in enumerate(names):
                means, errs = [], []
                for label in self.labels:
                    stats = self._stats(name, label, metric)
                    means.append(stats["mean"] if stats["mean"] is not None else np.nan)
                    errs.append(stats["std"] if stats["std"] is not None else 0.0)
                ax.bar(x + (i - (len(names) - 1) / 2) * width, means, width,
                       yerr=errs, color=colors[i], label=name, capsize=2)
            ax.set_xticks(x)
            ax.set_xticklabels([LABEL_NAMES[l] for l in self.labels])
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.2)
            if metric == "dice":
                ax.set_ylim(0, 1.05)
        fig.patch.set_facecolor("#1e1e1e")
        for ax in axes:
            ax.set_facecolor("#252525")
            ax.tick_params(colors="white")
            ax.yaxis.label.set_color("white")
            for spine in ax.spines.values():
                spine.set_color("#555555")
        axes[0].legend(facecolor="#2e2e2e", edgecolor="#555555",
                       labelcolor="white", fontsize=8)
        fig.suptitle("Per-class mean metrics", color="white", fontweight="bold")
        fig.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, facecolor=fig.get_facecolor(), dpi=150)
        print(f"saved -> {path}")
        plt.close(fig)
