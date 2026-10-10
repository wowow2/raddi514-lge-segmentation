"""Bad-performer markdown table: cases awful in ANY run, all labels."""

import os

import numpy as np

from src.config import LABEL_NAMES
from .cv_data import CVData


def worst_slice(gt: np.ndarray, pred: np.ndarray, label: int):
    """Slice with most GT voxels of `label` and zero pred overlap.

    Falls back to the slice with most GT voxels; None when GT has none.
    """
    counts = np.array([(gt[:, :, s] == label).sum() for s in range(gt.shape[2])])
    if counts.max() == 0:
        return None
    hit = np.array([((gt[:, :, s] == label) & (pred[:, :, s] == label)).sum()
                    for s in range(gt.shape[2])])
    cands = np.where((counts > 0) & (hit == 0))[0]
    return int(cands[np.argmax(counts[cands])]) if len(cands) else int(np.argmax(counts))


def fp_slice(pred: np.ndarray, label: int):
    """Slice with most predicted voxels of `label` (GT-empty cases)."""
    counts = [(pred[:, :, s] == label).sum() for s in range(pred.shape[2])]
    return int(np.argmax(counts)) if max(counts) > 0 else None


class BadPerformersTable:
    """Markdown table of cases with Dice < threshold on any label in any run."""

    def __init__(self, cv_data: CVData, labels: tuple[int, ...] = (1, 2, 3, 4),
                 awful_below: float = 0.2):
        self.cv = cv_data
        self.labels = labels
        self.awful_below = awful_below

    # ------------------------------------------------------------------- rows
    def rows(self):
        out = []
        for fi, nid, case in self.cv.iter_val():
            gt = self.cv.gt_array(case)
            entry = {"cid": case.case_id, "grp": case.group, "fold": fi,
                     "gt": {lbl: int((gt == lbl).sum()) for lbl in self.labels},
                     "dice": {}, "preds": {}}
            for name in self.cv.run_names:
                pred = self.cv.pred_array(name, fi, nid)
                if pred is None:
                    continue
                entry["dice"][name] = self.cv.case_dice(pred, gt)
                entry["preds"][name] = pred
            if any(dd[lbl] < self.awful_below
                   for dd in entry["dice"].values() for lbl in self.labels):
                out.append(entry)
        out.sort(key=lambda e: (min(dd[lbl] for dd in e["dice"].values()
                                    for lbl in self.labels), e["cid"]))
        return out

    # ----------------------------------------------------------------- render
    def _worst(self, entry):
        """(slice, tag) from the best run's worst *awful* label."""
        avail = [n for n in self.cv.run_names if n in entry["dice"]]
        best = avail[-1]
        dd = entry["dice"][best]
        pred = entry["preds"][best]
        gt_arr = self.cv.gt_array(
            self.cv.dataset.get(entry["cid"]))
        awful = [lbl for lbl in self.labels
                 if any(entry["dice"][n][lbl] < self.awful_below
                        for n in self.cv.run_names if n in entry["dice"])]
        lbl = min(awful, key=lambda l: dd[l])
        if entry["gt"][lbl] == 0:
            vols = {n: int((entry["preds"][n] == lbl).sum())
                    for n in self.cv.run_names if n in entry["preds"]}
            n = max(vols, key=vols.get)
            return fp_slice(entry["preds"][n], lbl), f"{LABEL_NAMES[lbl]}-fp"
        return worst_slice(gt_arr, pred, lbl), LABEL_NAMES[lbl]

    def to_markdown(self):
        short = {"plain": "pl", "resenc": "re", "regions": "rg",
                 "polar": "po"}
        dice_cols = " | ".join(
            f"{LABEL_NAMES[lbl]} ({'/'.join(short.get(n, n) for n in self.cv.run_names)})"
            for lbl in self.labels)
        gt_cols = " | ".join(f"GT {LABEL_NAMES[lbl]}" for lbl in self.labels)
        lines = [f"| Case | Group | Fold | {gt_cols} | {dice_cols} | Zero in | Worst sl. |",
                 "|" + "|".join(["---"] * (5 + 2 * len(self.labels))) + "|"]
        for e in self.rows():
            coldice = " | ".join(
                "/".join(f"{e['dice'][n][lbl]:.2f}" if n in e["dice"] else "—"
                         for n in self.cv.run_names)
                for lbl in self.labels)
            gtvol = " | ".join(str(e["gt"][lbl]) for lbl in self.labels)
            zero = sorted({n for n in self.cv.run_names if n in e["dice"]
                           if any(e["dice"][n][lbl] == 0.0 for lbl in self.labels)})
            ws, tag = self._worst(e)
            lines.append(
                f"| {e['cid']} | {e['grp']} | {e['fold']} | {gtvol} | {coldice} | "
                f"{','.join(zero) or '—'} | {ws} ({tag}) |")
        return "\n".join(lines) + "\n"

    def save(self, path: str):
        text = (f"# Bad performers (Dice < {self.awful_below} in any run, "
                f"post-processed)\n\n" + self.to_markdown())
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        print(text)
        print(f"saved -> {path}")
