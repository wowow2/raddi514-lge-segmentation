"""post-processing for EMIDEC predictions.
"""

import numpy as np

SCAR_LABEL = 3
MVO_LABEL = 4
MIN_SCAR_VOXELS = 50
MIN_MVO_VOXELS = 10


class PostProcessor:
    """Apply the locked post-processing rules to one prediction at a time."""

    def __init__(self, min_scar_voxels: int = MIN_SCAR_VOXELS,
                 min_mvo_voxels: int = MIN_MVO_VOXELS):
        self.min_scar_voxels = min_scar_voxels
        self.min_mvo_voxels = min_mvo_voxels

    def gate_small_scar(self, pred: np.ndarray) -> np.ndarray:
        """Zero scar predictions smaller than the configured threshold."""
        out = pred.copy()
        if int((out == SCAR_LABEL).sum()) < self.min_scar_voxels:
            out[out == SCAR_LABEL] = 0
        return out

    def gate_small_mvo(self, pred: np.ndarray) -> np.ndarray:
        """Zero MVO predictions smaller than the configured threshold.
        """
        out = pred.copy()
        if self.min_mvo_voxels > 0 and int((out == MVO_LABEL).sum()) < self.min_mvo_voxels:
            out[out == MVO_LABEL] = 0
        return out

    def enforce_containment(self, pred: np.ndarray) -> np.ndarray:
        """Zero MVO when no scar is predicted (anatomical invariant).
        """
        out = pred.copy()
        if int((out == SCAR_LABEL).sum()) == 0:
            out[out == MVO_LABEL] = 0
        return out

    def __call__(self, pred: np.ndarray) -> np.ndarray:
        """Apply all locked rules: size gates first, containment last."""
        return self.enforce_containment(
            self.gate_small_mvo(self.gate_small_scar(pred)))

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(min_scar_voxels={self.min_scar_voxels}, "
                f"min_mvo_voxels={self.min_mvo_voxels})")
