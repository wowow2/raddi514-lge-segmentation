"""Fold-wise ScarNet inference and full-volume merge."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import torch

from src.cascade.geometry import crop_array
from src.config import (
    DEFAULT_GT_ROOT,
    DEFAULT_SCARNET_MODEL_DIR,
    DEFAULT_SCARNET_OOF_PRED_DIR,
    DEFAULT_SCARNET_PRED_DIR,
    DEFAULT_SPLITS_PATH,
    nnunet_id_to_case_id,
    repo_path,
)
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.postprocess import PostProcessor
from src.datautils.predictions import CVPredictionIndex
from src.datautils.splits import CVSplits
from src.layout import artifact_id
from src.runs import RunManifest, read_manifest

from .checkpoint import load_model
from .merge import merge_slice_predictions, resize_probabilities
from .preprocess import prepare_image_slice, prepare_prior, prior_box

CaseCallback = Callable[[int, str, np.ndarray], None]


def _config_value(config: dict, key: str, default):
    data = config.get("data") or {}
    if data.get(key) is not None:
        return data[key]
    params = (config.get("model") or {}).get("params") or {}
    return params.get(key, default)


def prediction_manifest(result: dict, model_dir: str | Path,
                        pred_out: str | Path, stage1_dir: str | Path,
                        splits: CVSplits, gt_root: str | Path,
                        *, gate: PostProcessor | None = None,
                        inference: dict | None = None,
                        prob_out: str | Path | None = None) -> RunManifest:
    model_dir, pred_out = Path(model_dir), Path(pred_out)
    parent = RunManifest.from_dict(read_manifest(model_dir / "run.json"))
    data = result.get("data") or {}
    return RunManifest(
        run_id=artifact_id(pred_out),
        stage="stage2",
        description=parent.description,
        display_name=parent.display_name,
        tags=list(parent.tags),
        parent_run=parent.run_id or model_dir.name,
        source_run=artifact_id(stage1_dir),
        model=result["config"].get("model") or parent.model,
        training=parent.training,
        data={
            "gt_root": str(Path(gt_root).resolve()),
            "splits_path": str(splits.path),
            "splits_sha256": splits.sha256,
            "n_folds": len(splits.folds),
            "n_cases": len(splits.case_ids()),
            "stage1_predictions": str(Path(stage1_dir).resolve()),
            "out_size": data.get("out_size"),
            "pad": data.get("pad"),
            "min_size": data.get("min_size"),
            "multiple": data.get("multiple"),
            "clip_pct": data.get("clip_pct"),
            "use_largest_component": data.get("use_largest_component"),
        },
        inference={"checkpoint": "best", **(inference or {})},
        postprocess=({
            "ops": gate.ops(),
            "min_scar_voxels": gate.min_scar_voxels,
            "min_mvo_voxels": gate.min_mvo_voxels,
            "scar_grow_px": gate.scar_grow_px,
        } if gate is not None else {"ops": []}),
        paths={
            "predictions": str(pred_out.resolve()),
            "checkpoints": str(model_dir.resolve()),
            **({"probabilities": str(Path(prob_out).resolve())}
               if prob_out is not None else {}),
        },
        status={"expected_outputs": "oof", "prediction": result["prediction"]},
    )


@torch.no_grad()
def predict_oof(
    model_dir: str | Path = DEFAULT_SCARNET_MODEL_DIR,
    pred_out_dir: str | Path = DEFAULT_SCARNET_OOF_PRED_DIR,
    stage1_dir: str | Path = DEFAULT_SCARNET_PRED_DIR,
    splits: CVSplits | None = None,
    gt_root: str | Path = DEFAULT_GT_ROOT,
    *,
    gate: PostProcessor | None = None,
    device: str = "cpu",
    batch_size: int = 8,
    prob_out_dir: str | Path | None = None,
    on_case: CaseCallback | None = None,
) -> dict:
    """Run each fold's best checkpoint on that fold's validation cases."""
    model_dir = Path(repo_path(model_dir))
    pred_out = Path(repo_path(pred_out_dir))
    prob_out = Path(repo_path(prob_out_dir)) if prob_out_dir is not None else None
    stage1_dir = repo_path(stage1_dir)
    gt_root = repo_path(gt_root)
    splits = splits or CVSplits.load(DEFAULT_SPLITS_PATH)

    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA requested but not available")

    dataset = EMIDECDataset(gt_root)
    splits.validate(case_ids=dataset.case_ids).raise_if_invalid()
    stage1 = CVPredictionIndex(stage1_dir, splits)
    missing = stage1.missing()
    if missing:
        raise FileNotFoundError(
            f"missing stage-1 predictions: {', '.join(missing[:5])}")

    config: dict = {}
    written = 0
    out_size = pad = min_size = multiple = clip_pct = None
    use_largest_component = True

    for fold, split in enumerate(splits.folds):
        ckpt_path = model_dir / f"fold_{fold}" / "checkpoint_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for fold {fold}: {ckpt_path}")

        model, config = load_model(ckpt_path, device=device)
        out_size = int(_config_value(config, "out_size", 128))
        pad = int(_config_value(config, "pad", 16))
        min_size = int(_config_value(config, "min_size", 64))
        multiple = int(_config_value(config, "multiple", 16))
        clip_pct = tuple(_config_value(config, "clip_pct", (1.0, 99.0)))
        use_largest_component = bool(
            _config_value(config, "use_largest_component", True))

        fold_val_dir = pred_out / f"fold_{fold}" / "validation"
        fold_val_dir.mkdir(parents=True, exist_ok=True)
        fold_prob_dir = None
        if prob_out is not None:
            fold_prob_dir = prob_out / f"fold_{fold}" / "validation"
            fold_prob_dir.mkdir(parents=True, exist_ok=True)

        for nnunet_id in split["val"]:
            cid = nnunet_id_to_case_id(nnunet_id)
            case = dataset.get(cid)
            raw, _ = case.load()

            ref = nib.load(str(stage1.ref(cid).path))
            prior = prepare_prior(
                np.asanyarray(ref.dataobj).astype(np.uint8),
                use_largest_component=use_largest_component)
            box = prior_box(
                prior, pad=pad, min_size=min_size, multiple=multiple)
            crop = crop_array(raw, box)

            zs = [z for z in range(raw.shape[2]) if prior[:, :, z].any()]
            slice_preds: dict[int, np.ndarray] = {}
            slice_probs: dict[int, np.ndarray] = {}
            for start in range(0, len(zs), batch_size):
                chunk = zs[start:start + batch_size]
                images = [
                    prepare_image_slice(crop[:, :, z], out_size, clip_pct)
                    for z in chunk
                ]
                if not images:
                    continue
                batch = torch.stack(images).to(device)
                outputs = model(batch)
                logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                probs = logits.softmax(dim=1).cpu().numpy()
                pred = probs.argmax(axis=1)
                for i, z in enumerate(chunk):
                    slice_preds[z] = pred[i]
                    if prob_out is not None:
                        slice_probs[z] = probs[i]

            merged = merge_slice_predictions(slice_preds, box, raw.shape)
            if gate is not None:
                merged = gate(merged)

            nib.save(
                nib.Nifti1Image(merged.astype(np.uint8), ref.affine),
                str(fold_val_dir / f"{nnunet_id}.nii.gz"))
            if fold_prob_dir is not None:
                prob_vol = np.zeros(
                    (5, raw.shape[0], raw.shape[1], raw.shape[2]), np.float16)
                for z, probs in slice_probs.items():
                    resized = resize_probabilities(
                        probs, (box.width, box.height))
                    prob_vol[:, box.x0:box.x1, box.y0:box.y1, z] = resized
                np.savez_compressed(
                    fold_prob_dir / f"{nnunet_id}.npz",
                    probabilities=np.ascontiguousarray(
                        prob_vol.transpose(0, 3, 2, 1)))
            written += 1
            if on_case is not None:
                on_case(fold, cid, merged)

    return {
        "config": config,
        "data": {
            "out_size": out_size,
            "pad": pad,
            "min_size": min_size,
            "multiple": multiple,
            "clip_pct": list(clip_pct) if clip_pct is not None else None,
            "use_largest_component": use_largest_component,
        },
        "n_written": written,
        "prob_out": str(prob_out) if prob_out is not None else None,
        "prediction": CVPredictionIndex(pred_out, splits).summary(),
    }
