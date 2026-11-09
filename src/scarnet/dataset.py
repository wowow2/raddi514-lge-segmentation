"""2D ScarNet slices cropped to the 500over ROI prior."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from src.cascade.geometry import CropBox, crop_array
from src.config import (
    DEFAULT_GT_ROOT,
    DEFAULT_SCARNET_PRED_DIR,
    DEFAULT_SPLITS_PATH,
    repo_path,
)
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.predictions import CVPredictionIndex
from src.datautils.splits import CVSplits

from .preprocess import (
    prepare_image_slice,
    prepare_mask_slice,
    prepare_prior,
    prior_box,
)


class ScarNetDataset(Dataset):
    """One 2D LGE slice per sample, cropped to the 500over ROI prior."""

    def __init__(
        self,
        gt_root: str = DEFAULT_GT_ROOT,
        pred_dir: str | None = DEFAULT_SCARNET_PRED_DIR,
        splits_path: str = DEFAULT_SPLITS_PATH,
        fold: int = 0,
        split: str = "train",
        pad: int = 16,
        out_size: int = 128,
        clip_pct: tuple[float, float] = (1.0, 99.0),
        min_size: int = 64,
        multiple: int = 16,
        use_largest_component: bool = True,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if pred_dir is None:
            raise ValueError("pred_dir is required (the stage-1 prior defines the crop)")

        gt_root = str(repo_path(gt_root))
        pred_dir = str(repo_path(pred_dir))

        self.split = split
        self.fold = fold
        self.out_size = out_size
        self.pad = pad
        self.clip_pct = clip_pct
        self.min_size = min_size
        self.multiple = multiple
        self.use_largest_component = use_largest_component

        self.dataset = EMIDECDataset(gt_root)
        self.splits = CVSplits.load(splits_path)
        self.splits.validate().raise_if_invalid()

        self.case_ids = sorted(
            self.splits.train_ids(fold) if split == "train"
            else self.splits.val_ids(fold)
        )

        self.pred_index = CVPredictionIndex(pred_dir, splits=self.splits)
        missing = [cid for cid in self.case_ids if cid not in self.pred_index]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} case(s) have no stage-1 prediction in {pred_dir} "
                f"(first: {missing[0]})")
        absent = [cid for cid in self.case_ids if not self.pred_index.ref(cid).exists]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} stage-1 prediction file(s) are missing in {pred_dir} "
                f"(first: {absent[0]})")

        self.boxes: dict[str, CropBox] = {}
        self.samples: list[tuple[str, int]] = []
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._build_index()

    def _build_index(self) -> None:
        for cid in self.case_ids:
            image, gt = self.dataset.get(cid).load()
            self._cache[cid] = (image, gt)

            prior = self._stage1_mask(cid)
            self.boxes[cid] = prior_box(
                prior, pad=self.pad, min_size=self.min_size, multiple=self.multiple)

            slice_mask = (gt > 0) if self.split == "train" else prior
            for z in range(slice_mask.shape[2]):
                if slice_mask[:, :, z].any():
                    self.samples.append((cid, z))

    def _stage1_mask(self, case_id: str) -> np.ndarray:
        return prepare_prior(
            self.pred_index.load(case_id),
            use_largest_component=self.use_largest_component)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        cid, z = self.samples[idx]
        image, gt = self._cache[cid]
        box = self.boxes[cid]

        image_crop = crop_array(image, box)[:, :, z]
        mask_crop = crop_array(gt, box)[:, :, z]

        return {
            "image": prepare_image_slice(
                image_crop, self.out_size, self.clip_pct),
            "mask": prepare_mask_slice(mask_crop, self.out_size),
            "case_id": cid,
            "z": z,
            "box": box.as_dict(),
        }
