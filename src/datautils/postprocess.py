"""post-processing for EMIDEC predictions.
"""

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

SCAR_LABEL = 3
MVO_LABEL = 4
CAVITY_LABEL = 1
MYO_LABEL = 2
MIN_SCAR_VOXELS = 50
MIN_MVO_VOXELS = 10

# Slices are 10 mm thick, so scar growth stays in-plane: (X, Y, Z) arrays.
_INPLANE = ndi.generate_binary_structure(2, 1)[:, :, None]


class PostProcessor:
    """Apply the locked post-processing rules to one prediction at a time."""

    def __init__(self, min_scar_voxels: int = MIN_SCAR_VOXELS,
                 min_mvo_voxels: int = MIN_MVO_VOXELS,
                 scar_grow_px: int = 0):
        self.min_scar_voxels = min_scar_voxels
        self.min_mvo_voxels = min_mvo_voxels
        self.scar_grow_px = scar_grow_px

    def gate_small_scar(self, pred: np.ndarray) -> np.ndarray:
        """Zero scar predictions smaller than the configured threshold."""
        out = pred.copy()
        if int((out == SCAR_LABEL).sum()) < self.min_scar_voxels:
            out[out == SCAR_LABEL] = 0
        return out

    def grow_scar(self, pred: np.ndarray) -> np.ndarray:
        """Grow gated scar in-plane into cavity/myocardium only."""
        if self.scar_grow_px <= 0:
            return pred.copy()
        scar = pred == SCAR_LABEL
        if not scar.any():
            return pred.copy()
        out = pred.copy()
        grown = ndi.binary_dilation(
            scar, structure=_INPLANE, iterations=self.scar_grow_px)
        target = grown & np.isin(pred, (CAVITY_LABEL, MYO_LABEL))
        out[target] = SCAR_LABEL
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
        """Apply all locked rules: size gates first, containment last.

        The scar gate must precede growth: growing first lets sub-threshold
        specks pass the gate and raises normal-case false positives.
        """
        return self.enforce_containment(
            self.gate_small_mvo(self.grow_scar(self.gate_small_scar(pred))))

    def ops(self) -> list[dict]:
        """Ordered operations with their parameters (for run metadata)."""
        return [
            {"name": "gate_small_scar",
             "params": {"min_voxels": self.min_scar_voxels}},
            {"name": "grow_scar", "params": {"px": self.scar_grow_px}},
            {"name": "gate_small_mvo",
             "params": {"min_voxels": self.min_mvo_voxels}},
            {"name": "enforce_containment", "params": {}},
        ]

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(min_scar_voxels={self.min_scar_voxels}, "
                f"min_mvo_voxels={self.min_mvo_voxels}, "
                f"scar_grow_px={self.scar_grow_px})")


def apply_to_dir(src_dir: str | Path, dst_dir: str | Path,
                 post: PostProcessor) -> list[Path]:
    """Apply post-processing to every ``fold_*/validation/*.nii.gz`` tree.

    Preserves the directory layout and the source header/affine.
    """
    import nibabel as nib

    written: list[Path] = []
    for fold in sorted(Path(src_dir).glob("fold_*")):
        for src in sorted((fold / "validation").glob("*.nii.gz")):
            dst = Path(dst_dir) / fold.name / "validation" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = nib.load(str(src))
            cleaned = post(np.asanyarray(img.dataobj).astype(np.uint8))
            nib.save(nib.Nifti1Image(cleaned, img.affine, img.header), str(dst))
            written.append(dst)
    return written
