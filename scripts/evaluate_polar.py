"""Evaluate the polar cascade out-of-fold and record a run manifest.

Inference/merge lives in :func:`src.polar.predict.predict_oof`. By default the
merged predictions are gated in memory and the gated labels are saved (the
historical behaviour). Pass ``--raw`` to save ungated merged labels instead,
then use ``scripts/apply_postprocess.py`` to try different gates without
repeating inference.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.evaluator import DatasetEvaluator
from src.datautils.postprocess import MIN_MVO_VOXELS, MIN_SCAR_VOXELS, PostProcessor
from src.datautils.splits import CVSplits
from src.polar.predict import predict_oof
from src.runs import RunManifest, provenance_now


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Polar Cascade Cross-Validation")
    parser.add_argument("--model-dir", default="var/polar_models")
    parser.add_argument("--pred-out-dir", default="var/polar_preds")
    parser.add_argument("--stage1-dir", default="var/nnUNet_validation_501_pp")
    parser.add_argument("--out", default="var/results/polar_cv_metrics.json")
    parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--mode", choices=["erase", "additive"], default="erase")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--min-scar-voxels", type=int, default=MIN_SCAR_VOXELS)
    parser.add_argument("--min-mvo-voxels", type=int, default=MIN_MVO_VOXELS)
    parser.add_argument("--scar-grow-px", type=int, default=0)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--raw", action="store_true",
                        help="Save and score ungated merged labels")
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--parent-run", default=None)
    parser.add_argument("--no-manifest", action="store_true")
    return parser.parse_args(argv)


def _read_manifest(directory: Path) -> dict:
    path = directory / "run.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def main(argv=None) -> int:
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gate = None if (args.no_gate or args.raw) else PostProcessor(
        min_scar_voxels=args.min_scar_voxels,
        min_mvo_voxels=args.min_mvo_voxels,
        scar_grow_px=args.scar_grow_px)

    dataset = EMIDECDataset(args.gt_root)
    splits = CVSplits.load(args.splits)
    splits.validate(case_ids=dataset.case_ids).raise_if_invalid()

    model_dir = Path(args.model_dir)
    pred_out = Path(args.pred_out_dir)
    run_id = args.run_id or pred_out.name
    parent = _read_manifest(model_dir)

    print("=" * 70)
    print("POLAR CASCADE OUT-OF-FOLD EVALUATION")
    print(f"Models: {model_dir} | Mode: {args.mode} | dilation {args.dilation} "
          f"| tau {args.tau} | Gate: {'off' if gate is None else 'on'}")
    print(f"Output: {pred_out}")
    print("=" * 70)

    evaluator = DatasetEvaluator()

    def on_case(fold: int, case_id: str, merged) -> None:
        evaluator.add_emidec_case(
            case=dataset.get(case_id), pred=merged, metadata={"fold": fold})

    result = predict_oof(
        model_dir=model_dir, pred_out_dir=pred_out, stage1_dir=args.stage1_dir,
        splits=splits, gt_root=args.gt_root, dilation=args.dilation,
        mode=args.mode, prob_thresh=args.tau, gate=gate, device=device,
        on_case=on_case)

    config = result["config"]
    parent_manifest = RunManifest.from_dict(parent)
    manifest = RunManifest(
        run_id=run_id,
        description=args.description or parent_manifest.description,
        display_name=args.display_name or parent_manifest.display_name,
        tags=sorted(set(parent_manifest.tags) | set(args.tag)),
        stage="stage2",
        parent_run=args.parent_run or parent_manifest.run_id or model_dir.name,
        source_run=Path(args.stage1_dir).name,
        model=config.get("model") or parent_manifest.model,
        data={
            "gt_root": args.gt_root,
            "splits_path": args.splits,
            "splits_sha256": splits.sha256,
            "n_folds": len(splits.folds),
            "n_cases": len(splits.case_ids()),
            "stage1_predictions": str(Path(args.stage1_dir).resolve()),
            "R": result["R"], "Theta": result["Theta"],
        },
        training=parent_manifest.training,
        inference={"checkpoint": "best", "device": device,
                   "mode": args.mode, "dilation": args.dilation,
                   "tau": args.tau},
        merge={"mode": args.mode, "dilation": args.dilation,
               "prob_thresh": args.tau},
        postprocess={"ops": (gate.ops() if gate is not None else []),
                     "min_scar_voxels": (gate.min_scar_voxels if gate else None),
                     "min_mvo_voxels": (gate.min_mvo_voxels if gate else None),
                     "scar_grow_px": (gate.scar_grow_px if gate else None)},
        paths={"predictions": str(pred_out.resolve()),
               "checkpoints": str(model_dir.resolve()),
               "metrics": str(Path(args.out).resolve())},
        status={"expected_outputs": "oof",
                "prediction": result["prediction"]},
        provenance=provenance_now("scripts/evaluate_polar.py"),
    )
    report = evaluator.aggregate(run_id=run_id, manifest=manifest.to_dict())
    report.save_json(args.out)

    if not args.no_manifest:
        with open(pred_out / "run.json", "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    print("\n" + "=" * 50)
    print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 50)
    print(report.to_markdown())
    print(f"\nSaved metrics -> {args.out} ({report.overall['n_cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
