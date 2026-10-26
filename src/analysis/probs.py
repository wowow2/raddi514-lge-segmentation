"""Load out-of-fold nnU-Net probability maps for the cascade and viewer.

nnU-Net saves probabilities as (C, Z, Y, X); the EMIDEC loader gives (X, Y, Z).
The spatial transpose below was verified against the exported .nii.gz labels
(argmax match = 1.0), so ``load`` returns (C, X, Y, Z) aligned with the GT.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from src.config import (
    DEFAULT_SPLITS_PATH,
    repo_path,
)
from src.datautils.predictions import CVPredictionIndex
from src.datautils.splits import CVSplits

NPZ_TO_XYZ = (0, 3, 2, 1)  # (C, Z, Y, X) -> (C, X, Y, Z)


class ProbIndex:
    """case_id -> out-of-fold .npz path (<prob_dir>/fold_k/validation/<nid>.npz)."""

    def __init__(self, prob_dir: str, splits_path: str = DEFAULT_SPLITS_PATH):
        prob_dir = str(repo_path(prob_dir))
        splits_path = str(repo_path(splits_path))
        self.index = CVPredictionIndex(
            prob_dir, CVSplits.load(splits_path), suffix=".npz")
        self.paths: dict[str, Path] = {
            cid: ref.path for cid, ref in self.index.items().items()}

    def available(self) -> list[str]:
        """Case ids whose .npz exists (the run may be partial)."""
        return self.index.available()

    def load(self, case_id: str) -> np.ndarray:
        """Return (C, X, Y, Z) float32 probabilities aligned with the GT."""
        p = np.load(self.paths[case_id])["probabilities"]
        return np.ascontiguousarray(p.transpose(*NPZ_TO_XYZ), dtype=np.float32)

    def verify(self, case_id: str, prob: np.ndarray) -> float:
        """Fraction of voxels where argmax(prob) equals the exported .nii.gz label."""
        nii_path = self.paths[case_id].with_suffix(".nii.gz")
        lab = nib.load(str(nii_path)).get_fdata().astype(np.uint8)
        if lab.shape != prob.shape[1:]:
            return 0.0
        return float((prob.argmax(0) == lab).mean())
