"""Shared cross-validation data loader for plotter classes.

Loads the splits file and ground truth once, resolves prediction paths per
run through :class:`CVPredictionIndex`, and caches arrays. Plots can also be
filled from saved metrics instead of re-reading NIfTI volumes.
"""

import nibabel as nib
import numpy as np

from src.config import case_id_to_nnunet_id, nnunet_id_to_case_id
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.metrics import SegmentationMetrics
from src.datautils.predictions import CVPredictionIndex
from src.datautils.splits import CVSplits


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
        self.splits = CVSplits.load(splits_path)
        validation = self.splits.validate()
        validation.raise_if_invalid()
        self.fold_of = self.splits.fold_of_map()
        self.runs = list(runs)
        self.run_names = [n for n, _ in self.runs]
        self.indices = {
            name: CVPredictionIndex(pred_dir, self.splits)
            for name, pred_dir in self.runs
        }
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
            path = self.indices[run].path(nnunet_id_to_case_id(nnunet_id))
            self._pred_cache[key] = (
                nib.load(path).get_fdata().astype(np.uint8)
                if path.exists() else None)
        return self._pred_cache[key]

    def iter_val(self):
        """Yield (fold_idx, nnunet_id, case) over every validation case."""
        for fi, split in enumerate(self.splits.folds):
            for raw in split["val"]:
                yield (fi, case_id_to_nnunet_id(raw),
                       self.dataset.get(nnunet_id_to_case_id(raw)))

    def case_dice(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        return self.metric.dice(pred, gt)

    def missing(self) -> dict[str, list[str]]:
        """Cases without a prediction file, per run."""
        return {name: index.missing() for name, index in self.indices.items()}
