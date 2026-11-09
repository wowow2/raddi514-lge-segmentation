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
        plt.rcParams.update({"font.family": "serif", "font.size": 9})
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
        axes[0].legend(frameon=False, fontsize=7, ncol=2)
        fig.suptitle("Per-class mean metrics")
        fig.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved -> {path}")
        plt.close(fig)
