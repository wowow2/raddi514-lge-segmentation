"""Bad-performer markdown table: cases awful in ANY run, all labels."""

import os
import re

import numpy as np

from src.config import LABEL_NAMES
from .cv_data import CVData


def _abbreviations(names: list[str]) -> dict[str, str]:
    """Short, collision-free labels for markdown headers."""
    short: dict[str, str] = {}
    used: set[str] = set()
    for name in names:
        parts = [p for p in re.split(r"[_\-/]+", name) if p]
        base = "".join(p[:3] for p in parts)[:4] or name[:4]
        candidate, i = base, 2
        while candidate in used:
            candidate = f"{base}{i}"
            i += 1
        used.add(candidate)
        short[name] = candidate
    return short


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
    """Markdown table of cases with Dice < threshold on any label in any run.

    ``use_images=False`` (or loading from saved metrics) keeps the table
    usable without prediction volumes; the worst-slice column is then empty.
    """

    def __init__(self, cv_data: CVData | None, labels: tuple[int, ...] = (1, 2, 3, 4),
                 awful_below: float = 0.2, reference_run: str | None = None,
                 use_images: bool = True):
        self.cv = cv_data
        self.labels = labels
        self.awful_below = awful_below
        self.reference_run = reference_run
        self.use_images = use_images
        self._tables = None

    # ------------------------------------------------------------------- rows
    def load_results(self, tables: dict) -> None:
        self._tables = tables
        self.use_images = False

    def _rows_from_results(self) -> list[dict]:
        tables = self._tables or {}
        case_ids = sorted({cid for table in tables.values() for cid in table.case_ids()})
        out = []
        for cid in case_ids:
            entry = {"cid": cid, "grp": None, "fold": None,
                     "gt": {lbl: None for lbl in self.labels},
                     "dice": {}, "preds": {}}
            for name, table in tables.items():
                rows = [r for r in table.rows if r["case_id"] == cid]
                if not rows:
                    continue
                entry["dice"][name] = {r["class_id"]: r["dice"] for r in rows}
                entry["gt"] = {r["class_id"]: r.get("gt_voxels") for r in rows}
                entry["grp"] = rows[0].get("group") or entry["grp"]
                entry["fold"] = rows[0].get("fold") if entry["fold"] is None else entry["fold"]
            if any(dd[lbl] < self.awful_below
                   for dd in entry["dice"].values() for lbl in self.labels
                   if dd.get(lbl) is not None):
                out.append(entry)
        out.sort(key=lambda e: (min(
            dd[lbl] for dd in e["dice"].values() for lbl in self.labels
            if dd.get(lbl) is not None), e["cid"]))
        return out

    def rows(self):
        if self._tables is not None:
            return self._rows_from_results()
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
        """(slice, tag) from the reference run's worst *awful* label."""
        avail = [n for n in entry["dice"] if n in entry["preds"]]
        if not avail:
            # Saved-metrics mode: no volumes loaded, so no worst slice.
            label = min(
                (lbl for lbl in self.labels
                 if any(entry["dice"][n].get(lbl, 1.0) < self.awful_below
                        for n in entry["dice"])),
                default=self.labels[0])
            return None, LABEL_NAMES[label]
        best = self.reference_run if self.reference_run in avail else avail[-1]
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
        names = (list(self.cv.run_names) if self.cv is not None
                 else list(self._tables or {}))
        short = _abbreviations(names)
        dice_cols = " | ".join(
            f"{LABEL_NAMES[lbl]} ({'/'.join(short.get(n, n) for n in names)})"
            for lbl in self.labels)
        gt_cols = " | ".join(f"GT {LABEL_NAMES[lbl]}" for lbl in self.labels)
        lines = [f"| Case | Group | Fold | {gt_cols} | {dice_cols} | Zero in | Worst sl. |",
                 "|" + "|".join(["---"] * (5 + 2 * len(self.labels))) + "|"]
        for e in self.rows():
            coldice = " | ".join(
                "/".join(f"{e['dice'][n][lbl]:.2f}" if n in e["dice"] else "—"
                         for n in names)
                for lbl in self.labels)
            gtvol = " | ".join(str(e["gt"][lbl]) for lbl in self.labels)
            zero = sorted({n for n in names if n in e["dice"]
                           if any(e["dice"][n][lbl] == 0.0 for lbl in self.labels)})
            ws, tag = self._worst(e)
            worst = "—" if ws is None else f"{ws} ({tag})"
            lines.append(
                f"| {e['cid']} | {e['grp']} | {e['fold']} | {gtvol} | {coldice} | "
                f"{','.join(zero) or '—'} | {worst} |")
        return "\n".join(lines) + "\n"

    def save(self, path: str):
        text = (f"# Bad performers (Dice < {self.awful_below} in any run)\n\n"
                + self.to_markdown())
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        print(text)
        print(f"saved -> {path}")
