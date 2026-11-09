"""Qualitative percentile gallery: shared case per row, all methods per row."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import LABEL_NAMES
from .cv_data import CVData


class PercentileCaseGallery:
    """Rows are percentile levels of the mean Dice across methods; each row
    shows the same case for every method so anatomies line up."""

    def __init__(self, cv_data: CVData, labels: tuple[int, ...] = (3, 4),
                 percentiles: tuple[int, ...] = (100, 75, 50, 25)):
        self.cv = cv_data
        self.labels = labels
        self.percentiles = percentiles

    def _collect(self):
        # per label: {case_id: (case, fi, nid, {run_name: dice})}
        records: dict[int, dict[str, tuple]] = {lbl: {} for lbl in self.labels}
        for fi, nid, case in self.cv.iter_val():
            gt = self.cv.gt_array(case)
            per_run = {name: self.cv.case_dice(self.cv.pred_array(name, fi, nid), gt)
                       for name in self.cv.run_names
                       if self.cv.pred_array(name, fi, nid) is not None}
            for lbl in self.labels:
                if int((gt == lbl).sum()) == 0:
                    continue
                dices = {name: float(d[lbl]) for name, d in per_run.items()}
                records[lbl][case.case_id] = (case, fi, nid, dices)
        self._records = records

    def _pick(self, label: int, pct: int):
        items = list(self._records[label].values())
        if not items:
            return None
        items.sort(key=lambda item: float(np.mean(list(item[3].values()))))
        index = int(round(pct / 100 * (len(items) - 1)))
        return items[index]

    @staticmethod
    def _slice(gt, label):
        counts = (gt == label).sum(axis=(0, 1))
        return int(np.argmax(counts)) if counts.size else 0

    def save(self, label: int, path: str):
        if not getattr(self, "_records", None):
            self._collect()
        names = list(self.cv.run_names)
        if not names or not self._records.get(label):
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
        case_ids = []
        for row, pct in enumerate(self.percentiles):
            picked = self._pick(label, pct)
            if picked is None:
                for col in range(ncols):
                    axes[row, col].axis("off")
                case_ids.append("")
                continue
            case, fi, nid, dices = picked
            case_ids.append(case.case_id.replace("Case_", ""))
            _, gt = case.load()
            z = self._slice(gt, label)
            gt_sl = gt[:, :, z].T
            for col, name in enumerate(names):
                ax = axes[row, col]
                ax.set_xticks([])
                ax.set_yticks([])
                pred = self.cv.pred_array(name, fi, nid)
                sl = np.asarray(case.display_image[:, :, z]).T
                ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.contour(gt_sl == label, levels=[0.5],
                           colors=["#d62728"], linewidths=1.0)
                dice = dices.get(name)
                if pred is not None:
                    pred_sl = pred[:, :, z].T
                    if (pred_sl == label).any():
                        ax.contour(pred_sl == label, levels=[0.5],
                                   colors=["#2ca02c"], linewidths=1.0)
                ax.set_title(
                    f"{case.case_id.replace('Case_', '')}: "
                    + (f"{dice:.2f}" if dice is not None else "n/a"),
                    fontsize=6.5, pad=1.5)
                if col == 0:
                    ax.set_ylabel(f"{pct}th", fontsize=8)
        handles = [
            plt.Line2D([], [], color="#d62728", lw=1.2, label="Ground truth"),
            plt.Line2D([], [], color="#2ca02c", lw=1.2, label="Prediction"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   frameon=False, fontsize=8)
        fig.suptitle(f"{LABEL_NAMES[label]} cases by mean Dice percentile",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0.03, 1, 0.88))
        top = max(axes[0, col].get_position().y1 for col in range(ncols))
        for col, name in enumerate(names):
            pos = axes[0, col].get_position()
            fig.text((pos.x0 + pos.x1) / 2, top + 0.03, name,
                     ha="center", va="bottom", fontsize=7)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved -> {path}")
        plt.close(fig)
