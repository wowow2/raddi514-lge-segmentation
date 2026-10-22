"""
Feature extraction from nnU-Net predictions
"""

from dataclasses import dataclass
from pathlib import Path
import nibabel as nib
import numpy as np

from src.config import (
    DEFAULT_SPLITS_PATH,
    LABEL_CAVITY,
    LABEL_MVO,
    LABEL_MYOCARDIUM,
    LABEL_SCAR,
    nnunet_id_to_case_id,
)
from src.datautils.predictions import CVPredictionIndex
from src.datautils.splits import CVSplits

WALL_LABELS = (LABEL_MYOCARDIUM, LABEL_SCAR, LABEL_MVO)
RADIUS_PADDING = 1.15
RADIUS_PERCENTILE = 99.5

@dataclass
class AnchorData:
    centers: np.ndarray # shape (Z,2)
    radii: np.ndarray  # shape (Z,)
    myo_container: np.ndarray # shape (H,W,Z)

class CaseAnchors:
    def __init__(self, pred_path: Path):
        nii = nib.load(str(pred_path))
        self.pred_path = str(pred_path)
        self.pred_vol = nii.get_fdata().astype(np.uint8)
        self.zooms = nii.header.get_zooms()[:3]
        self.anchors = self._compute_anchors()

    def _compute_centers(self) -> np.ndarray:
        cavity = self.pred_vol == LABEL_CAVITY
        counts = cavity.sum(axis=(0,1))
        H, W, Z = self.pred_vol.shape

        # Sum over x and y
        sum_x = np.tensordot(cavity, np.arange(W), axes=([1], [0])).sum(axis=0)
        sum_y = np.tensordot(cavity, np.arange(H), axes=([0], [0])).sum(axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            xc = sum_x / counts
            yc = sum_y / counts

        # interpolate
        valid = ~np.isnan(xc)
        if not valid.any():
            raise ValueError(
                f"{self.pred_path}: no predicted cavity voxels; cannot anchor "
                "the polar transform")
        z_all = np.arange(Z)
        xc_interp = np.interp(z_all, z_all[valid], xc[valid])
        yc_interp = np.interp(z_all, z_all[valid], yc[valid])

        return np.column_stack([xc_interp, yc_interp])

    def _compute_radii(self, centers: np.ndarray) -> np.ndarray:
        H, W, Z = self.pred_vol.shape
        rmax = np.full(Z, np.nan)

        for z in range(Z):
            ys, xs = np.nonzero(np.isin(self.pred_vol[:,:,z], WALL_LABELS))
            if xs.size == 0:
                continue
            xc, yc = centers[z]
            dist = np.hypot(xs - xc, ys - yc)
            rmax[z] = np.percentile(dist, RADIUS_PERCENTILE) * RADIUS_PADDING

        # interpolate gappy slices
        valid = ~np.isnan(rmax)
        if not valid.any():
            raise ValueError(
                f"{self.pred_path}: no predicted myocardium/scar/MVO voxels; "
                "cannot anchor the polar transform")
        z_all = np.arange(Z)
        rmax_interp = np.interp(z_all, z_all[valid], rmax[valid])
        return rmax_interp

    def _get_myo_container(self) -> np.ndarray:
        return np.isin(self.pred_vol, WALL_LABELS)

    def _compute_anchors(self)-> AnchorData:
        centers = self._compute_centers()
        radii = self._compute_radii(centers)
        myo_container = self._get_myo_container()
        return AnchorData(centers=centers, radii=radii, myo_container=myo_container)

class AnchorIndex:
    def __init__(self, pred_dir: str, splits_path: str = DEFAULT_SPLITS_PATH):
        self.splits = CVSplits.load(splits_path)
        self.index = CVPredictionIndex(pred_dir, self.splits)
        self.case_to_path: dict[str, Path] = {
            cid: ref.path for cid, ref in self.index.items().items()
        }

    def get_path(self, case_id: str) -> Path:
        case_id = nnunet_id_to_case_id(case_id)
        if case_id not in self.case_to_path:
            raise KeyError(f"Case {case_id} does not exist")
        return self.case_to_path[case_id]

    def get(self, case_id: str) -> CaseAnchors:
        return CaseAnchors(pred_path=self.get_path(case_id))

