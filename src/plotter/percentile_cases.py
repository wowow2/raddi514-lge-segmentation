"""Qualitative percentile gallery: shared case per row, all methods per row."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb

from src.config import LABEL_NAMES
from .cv_data import CVData


class PercentileCaseGallery:
    """Each column ranks its own cases by Dice and shows that method's case
    at each percentile level, so every column is monotonic top to bottom."""

    def __init__(self, cv_data: CVData, labels: tuple[int, ...] = (3, 4),
                 percentiles: tuple[int, ...] = (100, 75, 50, 25)):
        self.cv = cv_data
        self.labels = labels
        self.percentiles = percentiles

    def _collect(self):
        # per label: {case_id: (case, fi, nid, z, {run_name: slice_dice})}
        records: dict[int, dict[str, tuple]] = {lbl: {} for lbl in self.labels}
        for fi, nid, case in self.cv.iter_val():
            gt = self.cv.gt_array(case)
            preds = {name: self.cv.pred_array(name, fi, nid)
                     for name in self.cv.run_names}
            for lbl in self.labels:
                counts = (gt == lbl).sum(axis=(0, 1))
                if not counts.size or counts.max() == 0:
                    continue
                z = int(np.argmax(counts))
                gt_sl = gt[:, :, z].T == lbl
                sdice = {}
                for name, pred in preds.items():
                    if pred is None:
                        continue
                    p_sl = pred[:, :, z].T == lbl
                    tot = gt_sl.sum() + p_sl.sum()
                    sdice[name] = 1.0 if tot == 0 else float(
                        2.0 * (gt_sl & p_sl).sum() / tot)
                records[lbl][case.case_id] = (case, fi, nid, z, sdice)
        self._records = records

    @staticmethod
    def _fill(ax, mask, color: str) -> None:
        rgba = (*to_rgb(color), 0.40)
        cmap = ListedColormap([(0.0, 0.0, 0.0, 0.0), rgba])
        ax.imshow(mask, cmap=cmap, origin="lower", vmin=0, vmax=1,
                  interpolation="nearest")

    def _pick(self, label: int, pct: int, name: str):
        items = [item for item in self._records[label].values() if name in item[4]]
        if not items:
            return None
        items.sort(key=lambda item: item[4][name])
        index = int(round(pct / 100 * (len(items) - 1)))
        return items[index]

    @staticmethod
    def _crop_box(sl, label_masks, margin=20, min_side=80):
        content = np.zeros(sl.shape, dtype=bool)
        for mask in label_masks:
            content |= mask
        if not content.any():
            content = sl > 0.05
        rows = np.any(content, axis=1)
        cols = np.any(content, axis=0)
        if not rows.any():
            return None
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        side = max(r1 - r0, c1 - c0) + 2 * margin
        side = max(side, min_side)
        rc = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        h, w = sl.shape
        half = side // 2
        r_start = max(0, min(rc - half, h - side))
        c_start = max(0, min(cc - half, w - side))
        return (slice(r_start, min(h, r_start + side)),
                slice(c_start, min(w, c_start + side)))

    @staticmethod
    def _pad_square(arrs):
        h, w = arrs[0].shape
        side = max(h, w)
        out = []
        for a in arrs:
            pt, pl = (side - h) // 2, (side - w) // 2
            out.append(np.pad(a, ((pt, side - h - pt), (pl, side - w - pl))))
        return out

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
                                 figsize=(2.5 * ncols, 2.5 * nrows),
                                 squeeze=False)
        for row, pct in enumerate(self.percentiles):
            for col, name in enumerate(names):
                ax = axes[row, col]
                ax.set_xticks([])
                ax.set_yticks([])
                picked = self._pick(label, pct, name)
                if picked is None:
                    ax.axis("off")
                    continue
                case, fi, nid, z, sdice = picked
                gt = self.cv.gt_array(case)
                gt_sl = gt[:, :, z].T
                base = np.asarray(case.display_image[:, :, z]).T
                pred = self.cv.pred_array(name, fi, nid)
                box = self._crop_box(
                    base, [gt_sl == label]
                    + ([pred[:, :, z].T == label] if pred is not None else []))
                sl, gt_c = self._pad_square(
                    [base[box] if box else base, gt_sl[box] == label])
                ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=1,
                          interpolation="nearest")
                self._fill(ax, gt_c, "#d62728")
                ax.contour(gt_c, levels=[0.5],
                           colors=["#d62728"], linewidths=0.7)
                dice = sdice.get(name)
                if pred is not None:
                    _, pred_c = self._pad_square(
                        [np.zeros_like(sl), pred[:, :, z].T[box] == label])
                    if pred_c.any():
                        self._fill(ax, pred_c, "#2ca02c")
                        ax.contour(pred_c, levels=[0.5],
                                   colors=["#2ca02c"], linewidths=0.7)
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
            plt.Rectangle((0, 0), 1, 1, facecolor="#d62728", alpha=0.40,
                          edgecolor="#d62728", label="Ground truth"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#2ca02c", alpha=0.40,
                          edgecolor="#2ca02c", label="Prediction"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   frameon=False, fontsize=8)
        fig.suptitle(f"{LABEL_NAMES[label]} slices by per-method slice Dice "
                     "percentile", fontsize=12)
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
