"""Reconstruct run manifests for historical experiments from on-disk artifacts.

Only facts derivable from directory names, paths, the splits file, and the
documented pipeline are written. Everything else stays empty, and every
manifest is marked ``provenance.legacy`` and ``provenance.reconstructed`` so
it is never mistaken for a manifest written at run time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH, REPO_ROOT
from src.datautils.postprocess import (
    MIN_MVO_VOXELS,
    MIN_SCAR_VOXELS,
    PostProcessor,
)
from src.datautils.predictions import CVPredictionIndex
from src.datautils.splits import CVSplits
from src.runs import RunStore

SPLITS_REL = "var/nnUNet_validation/splits_final.json"
RESULTS_REL = "var/results"

STAGE1_METRICS = {
    "nnUNet_validation": "nnunet_2d_cv_metrics.json",
    "nnUNet_validation_pp": "nnunet_2d_cv_metrics_pp.json",
    "nnUNet_validation_pp_grow1": "plain_grow1.json",
    "nnUNet_validation_500resenc": "nnunet_500resenc_cv_metrics.json",
    "nnUNet_validation_500resenc_pp": "nnunet_500resenc_cv_metrics_pp.json",
    "nnUNet_validation_500resenc_pp_grow1": "resenc_grow1.json",
    "nnUNet_validation_500over": "nnunet_500over_cv_metrics.json",
    "nnUNet_validation_500over_pp": "nnunet_500over_cv_metrics_pp.json",
    "nnUNet_validation_500over_pp_grow1": "over_grow1.json",
    "nnUNet_validation_501": "nnunet_501_cv_metrics.json",
    "nnUNet_validation_501_pp": "nnunet_501_cv_metrics_pp.json",
    "nnUNet_validation_501_pp_grow1": "regions_grow1.json",
    "nnUNet_validation_502": "nnunet_502_cv_metrics.json",
    "nnUNet_validation_502_pp": None,
    "nnUNet_validation_502_pp_grow1": "2.5d_grow1.json",
}

POLAR_METRICS = {
    "polar_preds": None,
    "polar_preds_erase": "polar_cv_metrics.json",
    "polar_preds_add": "polar_cv_metrics_add.json",
    "polar_preds_add_t0.5": "polar_cv_metrics_add_t0.5.json",
    "polar_preds_add_t0.7": "polar_cv_metrics_add_t0.7.json",
    "polar_preds_add_t0.9": "polar_cv_metrics_add_t0.9.json",
    "polar_preds_d0": "polar_cv_metrics_d0.json",
    "polar_preds_radius": "polar_cv_metrics_radius.json",
}

STAGE1_BASES = {
    "nnUNet_validation": {
        "dataset": "Dataset500_EMIDEC",
        "trainer": "nnUNetTrainer",
        "plans": "nnUNetPlans",
        "label_scheme": "flat",
        "channels": 1,
        "tags": ["baseline", "flat"],
    },
    "nnUNet_validation_500resenc": {
        "dataset": "Dataset500_EMIDEC",
        "trainer": "nnUNetTrainer",
        "plans": "nnUNetResEncUNetMPlans",
        "label_scheme": "flat",
        "channels": 1,
        "tags": ["resenc", "flat"],
    },
    "nnUNet_validation_500over": {
        "dataset": "Dataset500_EMIDEC",
        "trainer": "nnUNetTrainer_probabilisticOversampling",
        "plans": "nnUNetResEncUNetMPlans",
        "label_scheme": "flat",
        "channels": 1,
        "tags": ["resenc", "flat", "oversampling"],
    },
    "nnUNet_validation_501": {
        "dataset": "Dataset501_EMIDEC_Regions",
        "trainer": "nnUNetTrainer",
        "plans": "nnUNetResEncUNetMPlans",
        "label_scheme": "regions",
        "channels": 1,
        "tags": ["resenc", "regions"],
    },
    "nnUNet_validation_502": {
        "dataset": "Dataset502_EMIDEC_25DRegions",
        "trainer": "nnUNetTrainer",
        "plans": "nnUNetResEncUNetMPlans",
        "label_scheme": "regions",
        "channels": 3,
        "tags": ["resenc", "regions", "2.5d"],
    },
}

STAGE1_CHECKPOINTS = {
    "nnUNet_validation_500resenc":
        "var/nnUNet_results/Dataset500_EMIDEC/"
        "nnUNetTrainer__nnUNetResEncUNetMPlans__2d",
    "nnUNet_validation_500over":
        "var/nnUNet_results/Dataset500_EMIDEC/"
        "nnUNetTrainer_probabilisticOversampling__nnUNetResEncUNetMPlans__2d",
}


@dataclass
class ReconstructedRun:
    run_id: str
    pred_dir: Path
    stage: str = "stage1"
    metrics: Path | None = None
    checkpoints: Path | None = None
    parent_run: str | None = None
    source_run: str | None = None
    model: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    inference: dict = field(default_factory=dict)
    merge: dict = field(default_factory=dict)
    postprocess: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _base_name(name: str) -> str:
    for suffix in ("_pp_grow1", "_pp", "_grow1"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _is_prediction_dir(path: Path) -> bool:
    return next(path.glob("fold_*/validation/*.nii.gz"), None) is not None


def _postprocess(name: str) -> dict:
    if not (name.endswith("_pp") or name.endswith("_grow1")):
        return {}
    grow = 1 if name.endswith("_grow1") else 0
    post = PostProcessor(
        min_scar_voxels=MIN_SCAR_VOXELS,
        min_mvo_voxels=MIN_MVO_VOXELS,
        scar_grow_px=grow,
    )
    return {
        "ops": post.ops(),
        "min_scar_voxels": post.min_scar_voxels,
        "min_mvo_voxels": post.min_mvo_voxels,
        "scar_grow_px": post.scar_grow_px,
    }


def _stage1_run(pred_dir: Path, root: Path) -> ReconstructedRun:
    name = pred_dir.name
    base = _base_name(name)
    spec = STAGE1_BASES.get(base)
    if spec is None:
        model = {"name": "nnunet", "configuration": "2d"}
        dataset = None
        label_scheme = None
        channels = None
        tags = ["nnunet"]
    else:
        model = {
            "name": "nnunet",
            "trainer": spec["trainer"],
            "plans": spec["plans"],
            "configuration": "2d",
        }
        dataset = spec["dataset"]
        label_scheme = spec["label_scheme"]
        channels = spec["channels"]
        tags = ["nnunet", "stage1", *spec["tags"]]

    if name.endswith("_pp"):
        tags.append("pp")
    if name.endswith("_grow1"):
        tags.append("grow1")

    metrics_name = STAGE1_METRICS.get(name)
    metrics = root / RESULTS_REL / metrics_name if metrics_name else None
    checkpoints = STAGE1_CHECKPOINTS.get(base)
    checkpoints = root / checkpoints if checkpoints else None

    parent = _base_name(name) if name != base else None
    if name.endswith("_grow1") and parent and (root / "var" / f"{parent}_pp").exists():
        parent = f"{parent}_pp"

    data = {
        "gt_root": DEFAULT_GT_ROOT,
        "splits_path": SPLITS_REL,
    }
    if dataset:
        data["dataset"] = dataset
    if label_scheme:
        data["label_scheme"] = label_scheme
    if channels is not None:
        data["channels"] = channels

    return ReconstructedRun(
        run_id=name,
        pred_dir=pred_dir,
        stage="stage1",
        metrics=metrics,
        checkpoints=checkpoints,
        parent_run=parent,
        model=model,
        data=data,
        inference={"checkpoint": "best"},
        postprocess=_postprocess(name),
        tags=tags,
    )


def _polar_run(pred_dir: Path, root: Path) -> ReconstructedRun:
    name = pred_dir.name
    is_radius = "radius" in name
    mode = "additive" if "add" in name else "erase"
    tau = None
    for suffix, value in (("t0.5", 0.5), ("t0.7", 0.7), ("t0.9", 0.9)):
        if name.endswith(suffix):
            tau = value
    dilation = 0 if name in {"polar_preds", "polar_preds_d0",
                             "polar_preds_add_t0.7"} else None

    model = {
        "name": "polar_unet",
        "params": {
            "in_channels": 1,
            "num_classes": 5,
            "use_radius": is_radius,
            "deep_supervision": True,
        },
    }
    data = {
        "gt_root": DEFAULT_GT_ROOT,
        "splits_path": SPLITS_REL,
        "label_scheme": "flat",
        "channels": 1,
        "R": 64,
        "Theta": 256,
        "stage1_predictions": "var/nnUNet_validation_501_pp",
    }
    tags = ["polar", "stage2", mode]
    if tau is not None:
        tags.append(f"tau{tau}")
    if is_radius:
        tags.append("radius")

    metrics_name = POLAR_METRICS.get(name)
    metrics = root / RESULTS_REL / metrics_name if metrics_name else None
    checkpoints = root / ("var/polar_models_radius" if is_radius else "var/polar_models")
    if not checkpoints.exists():
        checkpoints = None

    extra = {}
    if name == "polar_preds":
        extra["alias_of"] = "polar_preds_add_t0.7"

    return ReconstructedRun(
        run_id=name,
        pred_dir=pred_dir,
        stage="stage2",
        metrics=metrics,
        checkpoints=checkpoints,
        source_run="nnUNet_validation_501_pp",
        model=model,
        data=data,
        inference={"checkpoint": "best", "mode": mode,
                   "tau": tau, "dilation": dilation},
        merge={"mode": mode, "prob_thresh": tau, "dilation": dilation},
        tags=tags,
        extra=extra,
    )


def build_plan(root: str | Path = REPO_ROOT) -> list[ReconstructedRun]:
    root = Path(root)
    var = root / "var"
    candidates = sorted(
        [p for p in var.glob("nnUNet_validation*") if _is_prediction_dir(p)]
        + [p for p in var.glob("polar_preds*") if _is_prediction_dir(p)]
    )
    plan = []
    for pred_dir in candidates:
        if pred_dir.name.startswith("polar_preds"):
            plan.append(_polar_run(pred_dir, root))
        else:
            plan.append(_stage1_run(pred_dir, root))
    return plan


def _prediction_status(run: ReconstructedRun, splits: CVSplits,
                       root: Path) -> dict:
    index = CVPredictionIndex(run.pred_dir, splits)
    summary = index.summary()
    summary["pred_dir"] = _rel(run.pred_dir, root)
    return summary


def manifest_fields(run: ReconstructedRun, splits: CVSplits,
                    root: str | Path = REPO_ROOT) -> dict:
    root = Path(root)
    sources = [run.pred_dir]
    if run.metrics is not None:
        sources.append(run.metrics)
    if run.checkpoints is not None:
        sources.append(run.checkpoints)

    data = dict(run.data)
    data.update({
        "splits_sha256": splits.sha256,
        "n_folds": len(splits.folds),
        "n_cases": len(splits.case_ids()),
    })

    paths = {
        "predictions": _rel(run.pred_dir, root),
        "metrics": _rel(run.metrics, root) if run.metrics is not None else None,
        "checkpoints": (_rel(run.checkpoints, root)
                        if run.checkpoints is not None else None),
    }

    provenance = {
        "legacy": True,
        "reconstructed": True,
        "created_at": _now(),
        "source": "src/rebuild.py",
        "sources": [_rel(path, root) for path in sources],
        "reconstructed_from": [
            "prediction/checkpoint directory names",
            "README.md and NOTES.md",
            "metrics file",
            "splits_final.json",
        ],
        "note": "Unknown fields are intentionally empty; this manifest was "
                "reconstructed from on-disk artifacts, not written at run time.",
    }

    return {
        "run_id": run.run_id,
        "stage": run.stage,
        "parent_run": run.parent_run,
        "source_run": run.source_run,
        "tags": sorted(set(run.tags)),
        "model": run.model,
        "data": data,
        "training": {},
        "inference": run.inference,
        "postprocess": run.postprocess,
        "merge": run.merge,
        "paths": paths,
        "status": {
            "expected_outputs": "oof",
            "prediction": _prediction_status(run, splits, root),
        },
        "provenance": provenance,
        **run.extra,
    }


def write_manifests(plan: list[ReconstructedRun], store: RunStore,
                    root: str | Path = REPO_ROOT,
                    force: bool = False) -> dict[str, list[str]]:
    root = Path(root)
    splits = CVSplits.load(str(root / SPLITS_REL))
    written, skipped = [], []
    for run in plan:
        target = store.registry_dir / run.run_id / "run.json"
        if not force and target.exists():
            try:
                existing = json.loads(target.read_text())
            except (OSError, json.JSONDecodeError):
                existing = {}
            if not existing.get("provenance", {}).get("legacy"):
                skipped.append(run.run_id)
                continue
        run_manifest = run.pred_dir / "run.json"
        if run_manifest.exists() and not force:
            try:
                existing = json.loads(run_manifest.read_text())
            except (OSError, json.JSONDecodeError):
                existing = {}
            if not existing.get("provenance", {}).get("legacy"):
                skipped.append(run.run_id)
                continue
        if force and target.exists():
            target.unlink()
        fields = manifest_fields(run, splits, root)
        fields.pop("run_id", None)
        store.register(run.run_id, **fields)
        written.append(run.run_id)
    return {"written": written, "skipped": skipped}


def rebuild_index(store: RunStore, out: str | Path) -> int:
    runs = store.discover(refresh=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump([run.manifest.to_dict() for run in runs], f, indent=2)
    return len(runs)
