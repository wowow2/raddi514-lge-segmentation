"""Qualitative percentile gallery: shared case per row, all methods per row."""

from __future__ import annotations

from pathlib import Path

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

    @staticmethod
    def _crop_box(sl, label_masks, margin=12):
        content = sl > 0.05
        for mask in label_masks:
            content |= mask
        rows = np.any(content, axis=1)
        cols = np.any(content, axis=0)
        if not rows.any():
            return None
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        side = max(r1 - r0, c1 - c0) + 2 * margin
        rc = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        h, w = sl.shape
        half = side // 2
        r_start = max(0, min(rc - half, h - side))
        c_start = max(0, min(cc - half, w - side))
        return (slice(r_start, min(h, r_start + side)),
                slice(c_start, min(w, c_start + side)))

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
                                 figsize=(2.4 * ncols, 2.5 * nrows),
                                 squeeze=False)
        for row, pct in enumerate(self.percentiles):
            picked = self._pick(label, pct)
            if picked is None:
                for col in range(ncols):
                    axes[row, col].axis("off")
                continue
            case, fi, nid, dices = picked
            _, gt = case.load()
            z = self._slice(gt, label)
            gt_sl = gt[:, :, z].T
            base = np.asarray(case.display_image[:, :, z]).T
            preds = {name: self.cv.pred_array(name, fi, nid) for name in names}
            box = self._crop_box(
                base, [gt_sl == label] + [preds[n][:, :, z].T == label
                                          for n in names if preds[n] is not None])
            for col, name in enumerate(names):
                ax = axes[row, col]
                ax.set_xticks([])
                ax.set_yticks([])
                sl = base[box] if box else base
                ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=1,
                          interpolation="nearest")
                ax.contour(gt_sl[box] == label, levels=[0.5],
                           colors=["#d62728"], linewidths=1.0)
                dice = dices.get(name)
                if preds[name] is not None:
                    pred_sl = preds[name][:, :, z].T[box]
                    if (pred_sl == label).any():
                        ax.contour(pred_sl == label, levels=[0.5],
                                   colors=["#2ca02c"], linewidths=1.0)
                ax.text(0.03, 0.97,
                        f"{case.case_id.replace('Case_', '')}: "
                        + (f"{dice:.2f}" if dice is not None else "n/a"),
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=7, color="white",
                        bbox=dict(facecolor="black", alpha=0.5,
                                  edgecolor="none", pad=1.0))
                if col == 0:
                    ax.set_ylabel(f"{pct}th", fontsize=9)
        handles = [
            plt.Line2D([], [], color="#d62728", lw=1.2, label="Ground truth"),
            plt.Line2D([], [], color="#2ca02c", lw=1.2, label="Prediction"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   frameon=False, fontsize=8)
        fig.suptitle(f"{LABEL_NAMES[label]} cases by mean Dice percentile",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0.015, 1, 0.92), w_pad=0.02, h_pad=0.08)
        top = max(axes[0, col].get_position().y1 for col in range(ncols))
        for col, name in enumerate(names):
            pos = axes[0, col].get_position()
            fig.text((pos.x0 + pos.x1) / 2, top + 0.012, name,
                     ha="center", va="bottom", fontsize=8.5)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        for out in (path, str(Path(path).with_suffix(".pdf"))):
            fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
        print(f"saved -> {path} (+.pdf)")
        plt.close(fig)
