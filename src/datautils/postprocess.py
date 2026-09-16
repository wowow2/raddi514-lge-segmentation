"""post-processing for EMIDEC predictions.
"""

import numpy as np

SCAR_LABEL = 3
MIN_SCAR_VOXELS = 50


class PostProcessor:
    """Apply the locked post-processing rules to one prediction at a time."""

    def __init__(self, min_scar_voxels: int = MIN_SCAR_VOXELS):
        self.min_scar_voxels = min_scar_voxels

    def gate_small_scar(self, pred: np.ndarray) -> np.ndarray:
        """Zero scar predictions smaller than the configured threshold."""
        out = pred.copy()
        if int((out == SCAR_LABEL).sum()) < self.min_scar_voxels:
            out[out == SCAR_LABEL] = 0
        return out

    def __call__(self, pred: np.ndarray) -> np.ndarray:
        """Apply all locked rules."""
        return self.gate_small_scar(pred)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(min_scar_voxels={self.min_scar_voxels})"
