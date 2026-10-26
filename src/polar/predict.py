"""Shared out-of-fold inference for the polar cascade.

Writes merged label maps to ``<pred_out_dir>/fold_<k>/validation/<id>.nii.gz``
and optionally reports each case through a callback so evaluation can reuse
the in-memory array instead of reloading the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import torch

from src.config import nnunet_id_to_case_id, repo_path
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.postprocess import PostProcessor
from src.datautils.predictions import CVPredictionIndex
from src.datautils.splits import CVSplits
from src.layout import artifact_id
from src.runs import RunManifest, read_manifest
from .anchors import CaseAnchors
from .checkpoint import load_model, polar_resolution
from .merge import PolarMerger

CaseCallback = Callable[[int, str, np.ndarray], None]


def prediction_manifest(result, model_dir, pred_out, stage1_dir, splits, gt_root,
                        *, inference, gate=None):
    """Shared output identity and lineage for prediction and evaluation CLIs."""
    model_dir, pred_out = Path(model_dir), Path(pred_out)
    parent = RunManifest.from_dict(read_manifest(model_dir / "run.json"))
    return RunManifest(
        run_id=artifact_id(pred_out), stage="stage2",
        description=parent.description, display_name=parent.display_name,
        tags=list(parent.tags), parent_run=parent.run_id or model_dir.name,
        source_run=artifact_id(stage1_dir),
        model=result["config"].get("model") or parent.model,
        training=parent.training,
        data={"gt_root": str(gt_root), "splits_path": str(splits.path),
              "splits_sha256": splits.sha256, "n_folds": len(splits.folds),
              "n_cases": len(splits.case_ids()),
              "stage1_predictions": str(Path(stage1_dir).resolve()),
              "R": result["R"], "Theta": result["Theta"]},
        inference={"checkpoint": "best", **inference},
        merge={"mode": inference["mode"], "dilation": inference["dilation"],
               "prob_thresh": inference["tau"]},
        postprocess=({"ops": gate.ops(), "min_scar_voxels": gate.min_scar_voxels,
                      "min_mvo_voxels": gate.min_mvo_voxels, "scar_grow_px": gate.scar_grow_px}
                     if gate is not None else {"ops": []}),
        paths={"predictions": str(pred_out.resolve()), "checkpoints": str(model_dir.resolve())},
        status={"expected_outputs": "oof", "prediction": result["prediction"]},
    )


def predict_oof(
    model_dir: str | Path,
    pred_out_dir: str | Path,
    stage1_dir: str | Path,
    splits: CVSplits,
    gt_root: str,
    *,
    dilation: int = 2,
    mode: str = "erase",
    prob_thresh: float | None = None,
    gate: PostProcessor | None = None,
    device: str = "cpu",
    on_case: CaseCallback | None = None,
) -> dict:
    """Run every fold's best checkpoint on that fold's validation cases."""
    model_dir = Path(repo_path(model_dir))
    pred_out = Path(repo_path(pred_out_dir))
    stage1_dir = repo_path(stage1_dir)
    gt_root = repo_path(gt_root)
    dataset = EMIDECDataset(gt_root)
    splits.validate(case_ids=dataset.case_ids).raise_if_invalid()
    stage1 = CVPredictionIndex(stage1_dir, splits)
    missing = stage1.missing()
    if missing:
        raise FileNotFoundError(
            f"missing stage-1 predictions: {', '.join(missing[:5])}")
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA requested but not available")

    config: dict = {}
    written = 0
    for fold, split in enumerate(splits.folds):
        ckpt_path = model_dir / f"fold_{fold}" / "checkpoint_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {ckpt_path}")

        model, config = load_model(ckpt_path, device=device)
        R, Theta = polar_resolution(config)
        merger = PolarMerger(
            model=model, dilation=dilation, gate=gate, device=device,
            mode=mode, prob_thresh=prob_thresh, R=R, Theta=Theta)

        fold_val_dir = pred_out / f"fold_{fold}" / "validation"
        fold_val_dir.mkdir(parents=True, exist_ok=True)
        for nnunet_id in split["val"]:
            cid = nnunet_id_to_case_id(nnunet_id)
            case = dataset.get(cid)
            raw_img, _ = case.load()
            anchors = CaseAnchors(stage1.path(cid))
            merged = merger(raw_vol=raw_img, nnunet_pred=anchors.pred_vol,
                            case_anchors=anchors, apply_gate=gate is not None)

            nii_ref = nib.load(str(anchors.pred_path))
            nib.save(
                nib.Nifti1Image(merged.astype(np.uint8), nii_ref.affine, nii_ref.header),
                str(fold_val_dir / f"{nnunet_id}.nii.gz"))
            written += 1
            if on_case is not None:
                on_case(fold, cid, merged)

    return {
        "config": config,
        "R": polar_resolution(config)[0],
        "Theta": polar_resolution(config)[1],
        "n_written": written,
        "prediction": CVPredictionIndex(pred_out, splits).summary(),
    }
