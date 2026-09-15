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


class NnunetExporter:
    """Convert EMIDEC cases to nnUNet raw layout.

    Usage:
        NnunetExporter(out_dir="nnUNet_raw").export(train_cases, test_cases)

    For region-based training pass ``labels=REGION_LABEL_NAMES`` and
    ``regions_class_order=REGION_CLASS_ORDER`` (with a distinct task_name
    such as ``Dataset501_EMIDEC_Regions``, since output paths don't encode
    the label definition).
    """

    def __init__(self, out_dir: str = "nnUNet_raw",
                 task_name: str = TASK_NAME, overwrite: bool = False,
                 labels: dict | None = None,
                 regions_class_order: tuple | None = None):
        self.out_dir = out_dir
        self.task_name = task_name
        self.overwrite = overwrite
        self.labels = dict(labels) if labels is not None else dict(LABEL_NAMES)
        self.regions_class_order = regions_class_order
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
    def _save_image(cls, src_path: str, dst_path: str) -> None:
        """Copy an image NIfTI: unscaled float32 data + spacing-aware affine."""
        nii = nib.load(src_path)
        data = nii.get_fdata(dtype=np.float32)
        nib.save(nib.Nifti1Image(data, cls._spaced_affine(nii)), dst_path)

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
            self._save_image(case.img_path,
                             os.path.join(images_tr, f"{nid}_0000.nii.gz"))
            self._save_label(case.gt_path,
                             os.path.join(labels_tr, f"{nid}.nii.gz"))
            training.append({"image": f"./imagesTr/{nid}_0000.nii.gz",
                             "label": f"./labelsTr/{nid}.nii.gz"})

        test_ids = []
        for case in test_cases:
            nid = self.sanitize_id(case.case_id)
            self._save_image(case.img_path,
                             os.path.join(images_ts, f"{nid}_0000.nii.gz"))
            test_ids.append(f"./imagesTs/{nid}_0000.nii.gz")

        generate_dataset_json(
            task_dir,
            channel_names={0: "LGE"},
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
                f"task_name={self.task_name!r})")
