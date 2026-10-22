import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH
from src.datautils.splits import CVSplits
from src.datautils.emidec_dataset import EMIDECDataset
from .anchors import AnchorIndex
from .transforms import PolarTransform

class PolarDataset(Dataset):
    def __init__(self, gt_root: str = DEFAULT_GT_ROOT, pred_dir: str = "var/nnUNet_validation_501_pp", splits_path: str = DEFAULT_SPLITS_PATH,
            fold: int = 0, split: str = "train",R: int = 64, Theta: int = 256):
        self.dataset = EMIDECDataset(gt_root)
        self.anchor_index = AnchorIndex(pred_dir=pred_dir, splits_path=splits_path)
        self.R = R
        self.Theta = Theta
        self.split = split
        self.splits_path = splits_path
        self.fold = fold
        self.splits = CVSplits.load(splits_path)
        self.splits.validate().raise_if_invalid()

        self.case_ids = self._flatten_case_ids()

        # Precompute one transform per case
        self.transforms: dict[str, PolarTransform] = {}
        self.samples: list[tuple[str, int]] = []
        self.skipped_slices = 0
        self._precompute_transforms()

        print(f"PolarDataset(split={split} fold={fold}): "
              f"{len(self.samples)} slices, skipped {self.skipped_slices} wall-less")

    def __len__(self) -> int:
        return len(self.samples)

    def _flatten_case_ids(self) -> list[str]:
        if self.split == "val":
            return sorted(self.splits.val_ids(self.fold))
        if self.split == "train":
            return sorted(self.splits.train_ids(self.fold))
        raise ValueError(f"split must be 'train' or 'val', got {self.split!r}")

    def _precompute_transforms(self):
        for cid in self.case_ids:
            case_anchors = self.anchor_index.get(cid)
            self.transforms[cid] = PolarTransform(
                case_anchors.anchors, self.R, self.Theta
            )
            wall_per_slice = case_anchors.anchors.myo_container.any(axis=(0, 1))
            case = self.dataset.get(cid)
            # Load lazily to get slice count
            num_slices = case.load()[0].shape[2]
            for z in range(num_slices):
                if z < len(wall_per_slice) and wall_per_slice[z]:
                    self.samples.append((cid, z))
                else:
                    self.skipped_slices += 1

    @staticmethod
    def normalize_slice(img_slice: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(img_slice, 1), np.percentile(img_slice, 99)
        clipped = np.clip(img_slice, p1, p99)
        denom = max(p99 - p1, 1e-5)
        return (clipped - p1) / denom

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        cid, z = self.samples[idx]
        case = self.dataset.get(cid)
        raw_vol, gt_vol = case.load()

        # Slice along Z axis
        vol_z = raw_vol[:, :, z]
        gt_z = gt_vol[:, :, z]

        # Normalize and convert to tensors
        norm_slice = torch.from_numpy(self.normalize_slice(vol_z)).float().unsqueeze(0)
        norm_gt = torch.from_numpy(gt_z).long()

        # Precomputed transform
        polar_transform = self.transforms[cid]

        # Unroll image (bilinear) and ground truth (nearest)
        polar_img = polar_transform.to_polar(norm_slice, z=z)
        polar_gt = polar_transform.to_polar(norm_gt, z=z, is_mask=True)

        return {
            "image": polar_img,  # Shape: (1, R, Theta), float32
            "target": polar_gt,  # Shape: (R, Theta), int64
            "case_id": cid,
            "slice_idx": z,
        }






