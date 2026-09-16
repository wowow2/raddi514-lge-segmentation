"""Export EMIDEC to nnUNetv2 raw format (original spacing/affine preserved).

Layout: <out>/Dataset500_EMIDEC/{imagesTr,labelsTr,imagesTs} + dataset.json.
The dataset.json is written with nnUNet's official ``generate_dataset_json``
(extra ``training``/``test`` file lists are passed through as kwargs).
"""

import os
import shutil

import nibabel as nib
import numpy as np
from nnunetv2.dataset_conversion.generate_dataset_json import (
    generate_dataset_json,
)

TASK_NAME = "Dataset500_EMIDEC"
LABEL_NAMES = {
    "background": 0,
    "cavity": 1,
    "myocardium": 2,
    "scar": 3,
    "mvo": 4,
}
REGION_LABEL_NAMES = {
    "background": 0,
    "whole_heart": (1, 2, 3, 4),
    "myocardium": (2, 3, 4),
    "infarction": (3, 4),
    "no_reflow": 4,
}
REGION_CLASS_ORDER = (1, 2, 3, 4)


def _count_regions(labels: dict) -> int:
    """Mirror nnUNet's region count: non-background entries (scalars or tuples)."""
    n = 0
    for k, r in labels.items():
        if k == "ignore":
            continue
        if (np.isscalar(r) and r == 0) or (
                isinstance(r, (tuple, list)) and len(set(r)) == 1 and set(r) == {0}):
            continue
        n += 1
    return n


class NeighborStacker:
    """Build 2.5D input channels [z-1, z, z+1] from a volume.

    Edge slices replicate (slice 0 stands in for z=-1, last slice for z=+1),
    so every position carries full context and nnU-Net needs no boundary
    special-casing. Pure array logic — no nnU-Net dependency.
    """

    CHANNEL_NAMES = {0: "LGE_below", 1: "LGE", 2: "LGE_above"}

    def build_channels(self, volume: np.ndarray) -> list[np.ndarray]:
        """Return [below, center, above] volumes with clamped indices."""
        z = volume.shape[2]
        s = np.arange(z)
        return [volume[:, :, np.clip(s - 1, 0, z - 1)],
                volume[:, :, s],
                volume[:, :, np.clip(s + 1, 0, z - 1)]]


class NnunetExporter:
    """Convert EMIDEC cases to nnUNet raw layout.

    Usage:
        NnunetExporter(out_dir="var/nnUNet_raw").export(train_cases, test_cases)

    For region-based training pass ``labels=REGION_LABEL_NAMES`` and
    ``regions_class_order=REGION_CLASS_ORDER`` (with a distinct task_name
    such as ``Dataset501_EMIDEC_Regions``, since output paths don't encode
    the label definition).

    For 2.5D context pass ``neighbor_stack=True`` (e.g. with region labels
    for a ``Dataset502_*`` task): each case is written as three channel
    files ``_0000/_0001/_0002`` (slice z-1/z/z+1, edge-replicated).
    """

    def __init__(self,                  out_dir: str = "var/nnUNet_raw",
                 task_name: str = TASK_NAME, overwrite: bool = False,
                 labels: dict | None = None,
                 regions_class_order: tuple | None = None,
                 neighbor_stack: bool = False):
        self.out_dir = out_dir
        self.task_name = task_name
        self.overwrite = overwrite
        self.labels = dict(labels) if labels is not None else dict(LABEL_NAMES)
        self.regions_class_order = regions_class_order
        self.neighbor_stack = neighbor_stack
        self.stacker = NeighborStacker() if neighbor_stack else None
        has_regions = any(isinstance(v, (tuple, list)) and len(v) > 1
                          for v in self.labels.values())
        if has_regions:
            if self.regions_class_order is None:
                raise ValueError(
                    "Region labels require regions_class_order "
                    "(nnUNet asserts this at load time)")
            if len(self.regions_class_order) != _count_regions(self.labels):
                raise ValueError(
                    f"regions_class_order has {len(self.regions_class_order)} entries "
                    f"but {_count_regions(self.labels)} non-background regions defined")

    @staticmethod
    def sanitize_id(case_id: str) -> str:
        """Unique, filesystem-safe nnUNet id (Case_P001 -> EMIDEC_P001)."""
        return "EMIDEC_" + case_id.replace("Case_", "")

    @staticmethod
    def _spaced_affine(nii: nib.Nifti1Image) -> np.ndarray:
        """Affine carrying the header pixdim spacing (source affines are ~identity).

        Scales each rotation column to the corresponding zoom, preserving
        orientation and translation.
        """
        zooms = [float(z) for z in nii.header.get_zooms()[:3]]
        affine = np.array(nii.affine, dtype=float)
        for i in range(3):
            col_norm = float(np.linalg.norm(affine[:3, i]))
            if col_norm > 0:
                affine[:3, i] *= zooms[i] / col_norm
        return affine

    @classmethod
    def _load_image(cls, src_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Unscaled float32 data + spacing-aware affine from a source NIfTI."""
        nii = nib.load(src_path)
        return nii.get_fdata(dtype=np.float32), cls._spaced_affine(nii)

    @classmethod
    def _write_image(cls, data: np.ndarray, affine: np.ndarray,
                     dst_path: str) -> None:
        nib.save(nib.Nifti1Image(data, affine), dst_path)

    @classmethod
    def _save_image(cls, src_path: str, dst_path: str) -> None:
        """Copy an image NIfTI: unscaled float32 data + spacing-aware affine."""
        data, affine = cls._load_image(src_path)
        cls._write_image(data, affine, dst_path)

    def _save_case_images(self, src_path: str, dst_dir: str, nid: str) -> None:
        """Write _0000 (single channel) or _0000/_0001/_0002 (neighbor stack)."""
        data, affine = self._load_image(src_path)
        channels = (self.stacker.build_channels(data) if self.stacker is not None
                    else [data])
        for i, ch in enumerate(channels):
            self._write_image(ch, affine, os.path.join(dst_dir, f"{nid}_{i:04d}.nii.gz"))

    @classmethod
    def _save_label(cls, src_path: str, dst_path: str) -> None:
        """Copy a GT NIfTI with exact integer labels + spacing-aware affine."""
        nii = nib.load(src_path)
        data = np.rint(nii.get_fdata()).astype(np.uint8)
        nib.save(nib.Nifti1Image(data, cls._spaced_affine(nii)), dst_path)

    def export(self, train_cases, test_cases) -> str:
        """Write train/test cases to nnUNet raw layout. Returns the task dir."""
        train_cases, test_cases = list(train_cases), list(test_cases)
        task_dir = os.path.join(self.out_dir, self.task_name)
        if os.path.exists(task_dir):
            if not self.overwrite:
                raise FileExistsError(f"{task_dir} exists (pass overwrite=True)")
            shutil.rmtree(task_dir)

        images_tr = os.path.join(task_dir, "imagesTr")
        labels_tr = os.path.join(task_dir, "labelsTr")
        images_ts = os.path.join(task_dir, "imagesTs")
        for d in (images_tr, labels_tr, images_ts):
            os.makedirs(d, exist_ok=True)

        training = []
        for case in train_cases:
            if not case.has_gt:
                raise ValueError(f"Train case {case.case_id} has no ground truth")
            nid = self.sanitize_id(case.case_id)
            self._save_case_images(case.img_path, images_tr, nid)
            self._save_label(case.gt_path,
                             os.path.join(labels_tr, f"{nid}.nii.gz"))
            training.append({"image": f"./imagesTr/{nid}_0000.nii.gz",
                             "label": f"./labelsTr/{nid}.nii.gz"})

        test_ids = []
        for case in test_cases:
            nid = self.sanitize_id(case.case_id)
            self._save_case_images(case.img_path, images_ts, nid)
            test_ids.append(f"./imagesTs/{nid}_0000.nii.gz")

        channel_names = (dict(NeighborStacker.CHANNEL_NAMES)
                         if self.stacker is not None else {0: "LGE"})
        generate_dataset_json(
            task_dir,
            channel_names=channel_names,
            labels={k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in self.labels.items()},
            num_training_cases=len(training),
            file_ending=".nii.gz",
            training=training,
            test=test_ids,
            **({"regions_class_order": list(self.regions_class_order)}
               if self.regions_class_order is not None else {}),
        )
        return task_dir

    def __repr__(self) -> str:
        return (f"NnunetExporter(out_dir={self.out_dir!r}, "
                f"task_name={self.task_name!r}, "
                f"neighbor_stack={self.stacker is not None})")
