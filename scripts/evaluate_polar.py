"""Evaluate the polar cascade out-of-fold and record a run manifest.

Inference/merge lives in :func:`src.polar.predict.predict_oof`. By default the
merged predictions are gated in memory and the gated labels are saved (the
historical behaviour). Pass ``--raw`` to save ungated merged labels instead,
then use ``scripts/apply_postprocess.py`` to try different gates without
repeating inference.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_POLAR_MODEL_DIR, DEFAULT_STAGE1_PRED_DIR, artifact_default, DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH, repo_path
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.evaluator import DatasetEvaluator
from src.datautils.postprocess import MIN_MVO_VOXELS, MIN_SCAR_VOXELS, PostProcessor
from src.datautils.splits import CVSplits
from src.polar.predict import predict_oof, prediction_manifest
from src.layout import artifact_id
from src.runs import provenance_now, write_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Polar Cascade Cross-Validation")
    parser.add_argument("--model-dir", default=DEFAULT_POLAR_MODEL_DIR)
    parser.add_argument("--pred-out-dir", default=artifact_default("polar_preds_erase", "predictions", "experiments://polar/outputs/polar_preds_erase/predictions"))
    parser.add_argument("--stage1-dir", default=DEFAULT_STAGE1_PRED_DIR)
    parser.add_argument("--out", default=artifact_default("polar_preds_erase", "metrics", "experiments://polar/outputs/polar_preds_erase/metrics.json"))
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


def main(argv=None) -> int:
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gate = None if (args.no_gate or args.raw) else PostProcessor(
        min_scar_voxels=args.min_scar_voxels,
        min_mvo_voxels=args.min_mvo_voxels,
        scar_grow_px=args.scar_grow_px)

    gt_root = repo_path(args.gt_root)
    splits_path = repo_path(args.splits)
    stage1_dir = repo_path(args.stage1_dir)
    out_path = repo_path(args.out)
    dataset = EMIDECDataset(gt_root)
    splits = CVSplits.load(splits_path)
    splits.validate(case_ids=dataset.case_ids).raise_if_invalid()

    model_dir = Path(repo_path(args.model_dir))
    pred_out = Path(repo_path(args.pred_out_dir))
    run_id = args.run_id or artifact_id(pred_out)

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
        model_dir=model_dir, pred_out_dir=pred_out, stage1_dir=stage1_dir,
        splits=splits, gt_root=gt_root, dilation=args.dilation,
        mode=args.mode, prob_thresh=args.tau, gate=gate, device=device,
        on_case=on_case)

    manifest = prediction_manifest(
        result, model_dir, pred_out, stage1_dir, splits, gt_root, gate=gate,
        inference={"device": device, "mode": args.mode, "dilation": args.dilation, "tau": args.tau})
    manifest.run_id = run_id
    manifest.description = args.description or manifest.description
    manifest.display_name = args.display_name or manifest.display_name
    manifest.tags = sorted(set(manifest.tags) | set(args.tag))
    manifest.parent_run = args.parent_run or manifest.parent_run
    manifest.paths["metrics"] = str(Path(out_path).resolve())
    manifest.provenance = provenance_now("scripts/evaluate_polar.py")
    report = evaluator.aggregate(run_id=run_id, manifest=manifest.to_dict())
    report.save_json(out_path)

    if not args.no_manifest:
        write_manifest(pred_out / "run.json", manifest)

    print("\n" + "=" * 50)
    print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 50)
    print(report.to_markdown())
    print(f"\nSaved metrics -> {out_path} ({report.overall['n_cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
