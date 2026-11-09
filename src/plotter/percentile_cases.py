"""Qualitative percentile gallery: image + GT/prediction contours."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import LABEL_NAMES
from .cv_data import CVData


class PercentileCaseGallery:
    def __init__(self, cv_data: CVData, labels: tuple[int, ...] = (3, 4),
                 percentiles: tuple[int, ...] = (100, 75, 50, 25, 0)):
        self.cv = cv_data
        self.labels = labels
        self.percentiles = percentiles
        self._scores: dict[str, dict[int, list[tuple[float, int, str, object]]]] = {}

    def _collect(self):
        scores = {name: {lbl: [] for lbl in self.labels}
                  for name in self.cv.run_names}
        for fi, nid, case in self.cv.iter_val():
            gt = self.cv.gt_array(case)
            for name in self.cv.run_names:
                pred = self.cv.pred_array(name, fi, nid)
                if pred is None:
                    continue
                dice = self.cv.case_dice(pred, gt)
                for lbl in self.labels:
                    if int((gt == lbl).sum()) == 0:
                        continue
                    scores[name][lbl].append(
                        (float(dice[lbl]), fi, nid, case))
        for name in scores:
            for lbl in scores[name]:
                scores[name][lbl].sort(key=lambda item: item[0])
        self._scores = scores

    def _pick(self, values, pct):
        if not values:
            return None
        index = int(round(pct / 100 * (len(values) - 1)))
        return values[index]

    @staticmethod
    def _slice(gt, label):
        counts = (gt == label).sum(axis=(0, 1))
        return int(np.argmax(counts)) if counts.size else 0

    def save(self, label: int, path: str):
        if not self._scores:
            self._collect()
        names = [n for n in self.cv.run_names
                 if self._scores.get(n, {}).get(label)]
        if not names:
            print(f"skipped -> {path} (no {LABEL_NAMES[label]} cases)")
            return
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 8,
        })
        nrows, ncols = len(self.percentiles), len(names)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(2.7 * ncols, 2.7 * nrows),
                                 squeeze=False)
        for row, pct in enumerate(self.percentiles):
            for col, name in enumerate(names):
                ax = axes[row, col]
                ax.set_xticks([])
                ax.set_yticks([])
                picked = self._pick(self._scores[name][label], pct)
                if picked is None:
                    ax.axis("off")
                    continue
                dice, fi, nid, case = picked
                _, gt = case.load()
                pred = self.cv.pred_array(name, fi, nid)
                z = self._slice(gt, label)
                sl = np.asarray(case.display_image[:, :, z]).T
                gt_sl = gt[:, :, z].T
                pred_sl = pred[:, :, z].T if pred is not None else None
                ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.contour(gt_sl == label, levels=[0.5],
                           colors=["#d62728"], linewidths=1.0)
                if pred_sl is not None and (pred_sl == label).any():
                    ax.contour(pred_sl == label, levels=[0.5],
                               colors=["#2ca02c"], linewidths=1.0)
                ax.set_title(f"{case.case_id.replace('Case_', '')}: {dice:.2f}",
                             fontsize=6.5, pad=1.5)
                if col == 0:
                    ax.set_ylabel(f"{pct}th", fontsize=8)
                if row == 0:
                    underscores = [i for i, ch in enumerate(name) if ch == "_"]
                    if underscores:
                        cut = min(underscores,
                                  key=lambda i: abs(i - len(name) / 2))
                        wrapped = name[:cut + 1] + "\n" + name[cut + 1:]
                    else:
                        wrapped = name
                    ax.text(0.5, 1.52, wrapped, transform=ax.transAxes,
                            ha="center", va="top", fontsize=7,
                            linespacing=1.0)
        handles = [
            plt.Line2D([], [], color="#d62728", lw=1.2, label="Ground truth"),
            plt.Line2D([], [], color="#2ca02c", lw=1.2, label="Prediction"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   frameon=False, fontsize=8)
        fig.suptitle(f"{LABEL_NAMES[label]} cases by Dice percentile",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0.03, 1, 0.88))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved -> {path}")
        plt.close(fig)
