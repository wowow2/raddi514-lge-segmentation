"""Reliability (calibration) curves for saved probability maps."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import LABEL_NAMES
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.splits import CVSplits


class ReliabilityPlot:
    def __init__(self, gt_root: str, splits_path: str,
                 runs: list[tuple[str, object]],
                 labels: tuple[int, ...] = (3, 4), n_bins: int = 10):
        self.dataset = EMIDECDataset(gt_root)
        self.splits = CVSplits.load(splits_path)
        self.runs = runs
        self.labels = labels
        self.n_bins = n_bins

    def collect(self):
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        inner = edges[1:-1]
        data = {
            name: {lbl: {"counts": np.zeros(self.n_bins),
                         "conf": np.zeros(self.n_bins),
                         "obs": np.zeros(self.n_bins),
                         "sq": 0.0, "n": 0}
                   for lbl in self.labels}
            for name, _ in self.runs
        }
        for split in self.splits.folds:
            for raw in split["val"]:
                cid = raw if raw.startswith("Case_") else "Case_" + raw.removeprefix("EMIDEC_")
                _, gt = self.dataset.get(cid).load()
                for name, index in self.runs:
                    probs = index.load(cid)
                    if probs.shape[1:] != gt.shape:
                        continue
                    for lbl in self.labels:
                        p = probs[lbl].reshape(-1)
                        y = (gt == lbl).reshape(-1).astype(np.float32)
                        idx = np.clip(np.digitize(p, inner, right=True),
                                      0, self.n_bins - 1)
                        acc = data[name][lbl]
                        acc["counts"] += np.bincount(idx, minlength=self.n_bins)
                        acc["conf"] += np.bincount(
                            idx, weights=p, minlength=self.n_bins)
                        acc["obs"] += np.bincount(
                            idx, weights=y, minlength=self.n_bins)
                        diff = p - y
                        acc["sq"] += float(np.dot(diff, diff))
                        acc["n"] += int(diff.size)
        return data

    def save(self, path: str):
        data = self.collect()
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.runs), 1)))
        fig, axes = plt.subplots(1, len(self.labels),
                                 figsize=(6.2 * len(self.labels), 5.2),
                                 squeeze=False)
        axes = axes[0]
        for ax, lbl in zip(axes, self.labels):
            for i, (name, _) in enumerate(self.runs):
                acc = data[name][lbl]
                mask = acc["counts"] > 0
                conf = np.divide(acc["conf"], acc["counts"],
                                 out=np.zeros_like(acc["conf"]),
                                 where=mask)
                obs = np.divide(acc["obs"], acc["counts"],
                                out=np.zeros_like(acc["obs"]),
                                where=mask)
                brier = acc["sq"] / acc["n"] if acc["n"] else np.nan
                ax.plot(conf[mask], obs[mask], "-o", color=colors[i],
                        markersize=4, lw=1.6,
                        label=f"{name} (Brier {brier:.4g})")
            ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Observed frequency")
            ax.set_title(f"{LABEL_NAMES[lbl]} reliability",
                         color="white", fontweight="bold")
            ax.grid(alpha=0.2)
        fig.patch.set_facecolor("#1e1e1e")
        for ax in axes:
            ax.set_facecolor("#252525")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            for spine in ax.spines.values():
                spine.set_color("#555555")
            ax.legend(facecolor="#2e2e2e", edgecolor="#555555",
                      labelcolor="white", fontsize=7)
        fig.suptitle("Reliability of predicted class probabilities",
                     color="white", fontweight="bold")
        fig.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, facecolor=fig.get_facecolor(), dpi=150)
        print(f"saved -> {path}")
        plt.close(fig)
