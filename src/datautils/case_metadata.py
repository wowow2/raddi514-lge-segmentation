"""Case-level dataset metadata, stored separately and joined by case id.

Results rows intentionally keep only what is needed for filtering; image
geometry, clinical metadata, and GT voxel counts live here so they are not
duplicated per run. Only NIfTI headers and GT masks are read -- image volumes
are not kept in memory.
"""

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.emidec_loader import N_LABELS

SCHEMA_VERSION = 1


def case_metadata(case) -> dict:
    """Describe one case without retaining its image array."""
    image = nib.load(case.img_path)
    shape = [int(v) for v in image.shape]
    spacing = [round(float(z), 3) for z in image.header.get_zooms()[:3]]

    gt_voxels = {lbl: 0 for lbl in range(N_LABELS)}
    if case.has_gt:
        gt = np.asanyarray(nib.load(case.gt_path).dataobj)
        for lbl in range(N_LABELS):
            gt_voxels[lbl] = int(np.count_nonzero(gt == lbl))

    return {
        "case_id": case.case_id,
        "group": case.group,
        "has_gt": case.has_gt,
        "shape": shape,
        "spacing": spacing,
        "n_slices": shape[2],
        "gt_voxels": gt_voxels,
        "metadata": dict(case.clinical_meta),
        "metadata_path": case.meta_path,
    }


def collect_case_metadata(dataset: EMIDECDataset) -> list[dict]:
    return [case_metadata(case) for case in dataset.cases]


def save_case_metadata(path: str | Path, dataset: EMIDECDataset) -> str:
    path = Path(path)
    os.makedirs(path.parent or Path("."), exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION,
               "cases": collect_case_metadata(dataset)}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def load_case_metadata(path: str | Path) -> dict[str, dict]:
    with open(path) as f:
        payload = json.load(f)
    cases = payload.get("cases", payload if isinstance(payload, list) else [])
    return {c["case_id"]: c for c in cases}
