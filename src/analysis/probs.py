"""Load nnU-Net softmax maps (.npz) and compute scar-vs-cavity diagnostics.

nnU-Net saves probabilities as (C, Z, Y, X); the EMIDEC loader gives (X, Y, Z).
The spatial transpose below was verified against the exported .nii.gz labels
(argmax match = 1.0), so ``load`` returns (C, X, Y, Z) aligned with the GT.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from src.config import (
    DEFAULT_SPLITS_PATH,
    LABEL_CAVITY,
    LABEL_MYOCARDIUM,
    LABEL_SCAR,
)
from src.datautils.predictions import CVPredictionIndex
from src.datautils.splits import CVSplits

NPZ_TO_XYZ = (0, 3, 2, 1)  # (C, Z, Y, X) -> (C, X, Y, Z)
DEFAULT_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


class ProbIndex:
    """case_id -> out-of-fold .npz path (<prob_dir>/fold_k/validation/<nid>.npz)."""

    def __init__(self, prob_dir: str, splits_path: str = DEFAULT_SPLITS_PATH):
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


def predict_with_scar_threshold(prob: np.ndarray, t: float | None) -> np.ndarray:
    """Argmax labels; if t is given, also call scar wherever P(scar) > t.

    Only cavity and myocardium voxels are overridden (MVO and background are
    left alone), which is the question for cavity-absorbed scar.
    """
    pred = prob.argmax(0).astype(np.uint8)
    if t is not None:
        override = (prob[LABEL_SCAR] > t) & np.isin(pred, (LABEL_CAVITY, LABEL_MYOCARDIUM))
        pred[override] = LABEL_SCAR
    return pred


def scar_diagnostics(
    prob: np.ndarray, gt: np.ndarray, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
) -> dict:
    """Per-case numbers for the 'torn or confidently wrong?' question."""
    pred = prob.argmax(0)
    scar = gt == LABEL_SCAR
    out: dict = {"gt_scar": int(scar.sum())}
    if not scar.any():
        return out

    out["mean_p_scar_in_gt_scar"] = float(prob[LABEL_SCAR][scar].mean())
    out["mean_p_cav_in_gt_scar"] = float(prob[LABEL_CAVITY][scar].mean())
    out["frac_gt_scar_pred_scar"] = float((pred[scar] == LABEL_SCAR).mean())
    out["frac_gt_scar_pred_cav"] = float((pred[scar] == LABEL_CAVITY).mean())
    out["frac_gt_scar_pred_myo"] = float((pred[scar] == LABEL_MYOCARDIUM).mean())

    lost = scar & (pred == LABEL_CAVITY)  # scar voxels absorbed into the cavity
    out["n_lost_to_cavity"] = int(lost.sum())
    if lost.any():
        out["mean_p_scar_in_lost"] = float(prob[LABEL_SCAR][lost].mean())
        out["mean_p_cav_in_lost"] = float(prob[LABEL_CAVITY][lost].mean())
        out["median_p_scar_in_lost"] = float(np.median(prob[LABEL_SCAR][lost]))

    for t in thresholds:  # recall of GT scar if scar fired at P(scar) > t
        out[f"recall_at_{t}"] = float((prob[LABEL_SCAR][scar] > t).mean())
    return out