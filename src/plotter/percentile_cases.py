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
                 percentiles: tuple[int, ...] = (0, 25, 50, 75, 100)):
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
        ncols = len(self.percentiles)
        fig, axes = plt.subplots(len(names), ncols,
                                 figsize=(3.2 * ncols, 3.25 * len(names)),
                                 squeeze=False)
        for row, name in enumerate(names):
            for col, pct in enumerate(self.percentiles):
                ax = axes[row, col]
                ax.set_facecolor("#111111")
                ax.set_xticks([])
                ax.set_yticks([])
                picked = self._pick(self._scores[name][label], pct)
                if picked is None:
                    ax.axis("off")
                    continue
                dice, fi, nid, case = picked
                image, gt = case.load()
                pred = self.cv.pred_array(name, fi, nid)
                z = self._slice(gt, label)
                sl = np.asarray(case.display_image[:, :, z]).T
                gt_sl = gt[:, :, z].T
                pred_sl = pred[:, :, z].T if pred is not None else None
                ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.contour(gt_sl == label, levels=[0.5],
                           colors=["#3dff8f"], linewidths=0.9)
                if pred_sl is not None and (pred_sl == label).any():
                    ax.contour(pred_sl == label, levels=[0.5],
                               colors=["#ff4d4d"], linewidths=0.9)
                ax.set_title(f"{case.case_id.replace('Case_', '')}  "
                             f"Dice {dice:.2f}", color="white", fontsize=8)
                if row == 0:
                    ax.text(0.5, 1.10, f"{pct}th percentile",
                            transform=ax.transAxes, ha="center",
                            color="white", fontsize=9)
            axes[row, 0].set_ylabel(name, color="white", fontsize=9)
        handles = [
            plt.Line2D([], [], color="#3dff8f", lw=1.2, label="GT"),
            plt.Line2D([], [], color="#ff4d4d", lw=1.2, label="Prediction"),
        ]
        fig.legend(handles=handles, loc="upper right",
                   facecolor="#2e2e2e", edgecolor="#555555",
                   labelcolor="white")
        fig.patch.set_facecolor("#1e1e1e")
        fig.suptitle(f"{LABEL_NAMES[label]} percentile cases",
                     color="white", fontweight="bold")
        fig.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, facecolor=fig.get_facecolor(), dpi=150)
        print(f"saved -> {path}")
        plt.close(fig)
