"""Shared cross-validation data loader for plotter classes.

Loads the splits file and ground truth once, resolves prediction paths per
run, and caches arrays. All plotter classes consume this instead of
re-implementing the fold/val/pred plumbing.
"""

import json
import os

import nibabel as nib
import numpy as np

from src.config import nnunet_id_to_case_id
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.metrics import SegmentationMetrics


class CVData:
    """Ground truth + predictions for a set of CV runs sharing one splits file."""

    def __init__(self, gt_root: str, splits_path: str,
                 runs: list[tuple[str, str]]):
        """
        Args:
            gt_root: EMIDEC train root (has Contours/).
            splits_path: splits_final.json shared by all runs.
            runs: [(run_name, pred_dir)] where pred_dir holds
                fold_<i>/validation/<nnunet_id>.nii.gz.
        """
        self.dataset = EMIDECDataset(gt_root)
        with open(splits_path) as f:
            self.folds = json.load(f)
        self.fold_of = {nid: fi for fi, s in enumerate(self.folds)
                        for nid in s["val"]}
        self.runs = list(runs)
        self.run_names = [n for n, _ in self.runs]
        self.metric = SegmentationMetrics()
        self._gt_cache: dict = {}
        self._pred_cache: dict = {}

    # ------------------------------------------------------------- accessors
    def gt_array(self, case):
        if case.case_id not in self._gt_cache:
            self._gt_cache[case.case_id] = (
                nib.load(case.gt_path).get_fdata().astype(np.uint8))
        return self._gt_cache[case.case_id]

    def pred_array(self, run: str, fold_idx: int, nnunet_id: str):
        key = (run, fold_idx, nnunet_id)
        if key not in self._pred_cache:
            d = dict(self.runs)[run]
            p = os.path.join(d, f"fold_{fold_idx}", "validation",
                             f"{nnunet_id}.nii.gz")
            self._pred_cache[key] = (
                nib.load(p).get_fdata().astype(np.uint8)
                if os.path.exists(p) else None)
        return self._pred_cache[key]

    def iter_val(self):
        """Yield (fold_idx, nnunet_id, case) over every validation case."""
        for fi, split in enumerate(self.folds):
            for nid in split["val"]:
                yield fi, nid, self.dataset.get(nnunet_id_to_case_id(nid))

    def case_dice(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        return self.metric.dice(pred, gt)
