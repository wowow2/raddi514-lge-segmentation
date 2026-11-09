"""Dice reliability curves: R(d) = P(Dice > d)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import LABEL_NAMES
from src.results import ResultsTable


class ReliabilityPlot:
    def __init__(self, tables: dict[str, ResultsTable],
                 labels: tuple[int, ...] = (1, 2, 3, 4),
                 thresholds: int = 101):
        self.tables = tables
        self.labels = labels
        self.thresholds = thresholds

    def _values(self, name: str, label: int) -> list[float]:
        table = self.tables[name].filter(class_id=label)
        if label in (3, 4):
            table = table.filter(gt_present=True)
        values = [float(r["dice"]) for r in table.rows
                  if r.get("dice_defined") and r.get("dice") is not None]
        return values

    def save(self, path: str):
        plt.rcParams.update({"font.family": "serif", "font.size": 9})
        names = list(self.tables)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 1)))
        grid = np.linspace(0.0, 1.0, self.thresholds)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True,
                                 sharey=True)
        axes = np.ravel(axes)
        for ax, label in zip(axes, self.labels):
            for i, name in enumerate(names):
                values = np.asarray(self._values(name, label), dtype=float)
                if values.size == 0:
                    continue
                reliability = np.array([(values > d).mean() for d in grid])
                ax.plot(grid, reliability, color=colors[i], lw=1.4,
                        label=name)
            title = LABEL_NAMES[label]
            if label in (3, 4):
                title += " (GT-present)"
            ax.set_title(title)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.set_xlabel("Dice")
            ax.set_ylabel("Reliability  R(d) = P(Dice > d)")
        for ax in axes[len(self.labels):]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(names)),
                   frameon=False, fontsize=8)
        fig.suptitle("Reliability curves")
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved -> {path}")
        plt.close(fig)
