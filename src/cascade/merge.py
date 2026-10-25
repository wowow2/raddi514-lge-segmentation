"""Paste ROI-cascade predictions back into full-volume label space."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from src.config import case_id_to_nnunet_id, nnunet_id_to_case_id


@dataclass(frozen=True)
class MergeResult:
    case_id: str
    box: dict
    pred_shape: list[int]
    full_shape: list[int]
    nnunet_id: str


def load_manifest(manifest_path: str | Path) -> dict[str, dict]:
    """Map case_id -> bbox record from a cascade export manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    boxes = {}
    for entry in manifest.get("cases", []):
        case_id = entry["case_id"]
        if not case_id.startswith("Case_"):
            raise ValueError(f"expected Case_* id in manifest, got {case_id}")
        boxes[case_id] = entry["bbox"]
    if not boxes:
        raise ValueError(f"no cases in {manifest_path}")
    return boxes


def merge_case(pred_path: str | Path, bbox: dict, reference: nib.Nifti1Image) -> nib.Nifti1Image:
    """Paste one (X, Y, Z) ROI prediction into a zero full-volume label map."""
    pred = nib.load(pred_path)
    pred_arr = np.asarray(pred.dataobj)
    if pred_arr.ndim != 3:
        raise ValueError(f"expected (X, Y, Z) prediction, got {pred_arr.shape} in {pred_path}")

    full_shape = reference.shape[:3]
    x0, x1 = int(bbox["x0"]), int(bbox["x1"])
    y0, y1 = int(bbox["y0"]), int(bbox["y1"])
    expected = (x1 - x0, y1 - y0, full_shape[2])
    if pred_arr.shape != expected:
        raise ValueError(
            f"prediction {pred_path.name}: shape {pred_arr.shape} "
            f"does not match bbox crop of full volume {expected}"
        )

    full = np.zeros(full_shape, dtype=pred_arr.dtype)
    full[x0:x1, y0:y1, :] = pred_arr
    return nib.Nifti1Image(full, reference.affine, reference.header)


def merge_oof(pred_dir: str | Path, manifest_path: str | Path, gt_root: str | Path,
              out_dir: str | Path) -> list[MergeResult]:
    """Merge each fold/ validation prediction into full-volume space."""
    pred_dir = Path(pred_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boxes = load_manifest(manifest_path)
    results: list[MergeResult] = []
    for fold_dir in sorted(pred_dir.glob("fold_*")):
        out_fold = out_dir / fold_dir.name / "validation"
        out_fold.mkdir(parents=True, exist_ok=True)
        for pred_path in sorted((fold_dir / "validation").glob("*.nii.gz")):
            case_id = nnunet_id_to_case_id(pred_path.name.removesuffix(".nii.gz"))
            if case_id not in boxes:
                raise ValueError(f"case {case_id} has no bbox in {manifest_path}")
            reference = nib.load(_reference_path(gt_root, case_id))
            fused = merge_case(pred_path, boxes[case_id], reference)
            out_path = out_fold / f"{case_id_to_nnunet_id(case_id)}.nii.gz"
            nib.save(fused, out_path)
            results.append(MergeResult(
                case_id=case_id, box=boxes[case_id],
                pred_shape=list(fused.get_fdata().shape),  # noqa: PLR2004
                full_shape=list(reference.shape[:3]),
                nnunet_id=pred_path.name.removesuffix(".nii.gz"),
            ))
    return results


def _reference_path(gt_root: str | Path, case_id: str) -> Path:
    image = next(
        p for p in (Path(gt_root) / case_id / "Images").glob("*.nii.gz")
    )
    return image
