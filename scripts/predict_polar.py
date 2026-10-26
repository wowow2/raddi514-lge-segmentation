"""Run polar out-of-fold inference/merge and save ungated labels.

This deliberately does not post-process or score. Downstream::

    python scripts/apply_postprocess.py --pred-dir experiments/polar/outputs/polar_preds_raw/predictions \
        --out-dir experiments/polar/outputs/polar_preds_erase/predictions --min-scar-voxels 50 --scar-grow-px 0
    python scripts/evaluate_nnunet_cv.py --pred-dir experiments/polar/outputs/polar_preds_erase/predictions \
        --out experiments/polar/outputs/polar_preds_erase/metrics.json

Changing gates or scar growth then never requires another forward pass.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (DEFAULT_POLAR_MODEL_DIR, DEFAULT_STAGE1_PRED_DIR, DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH, repo_path)
from src.datautils.splits import CVSplits
from src.polar.predict import predict_oof, prediction_manifest
from src.layout import artifact_id
from src.runs import provenance_now, write_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Polar cascade prediction (no scoring)")
    parser.add_argument("--model-dir", default=DEFAULT_POLAR_MODEL_DIR)
    parser.add_argument("--out-dir", default=str(repo_path("experiments://polar/outputs/polar_preds_raw/predictions")))
    parser.add_argument("--stage1-dir", default=DEFAULT_STAGE1_PRED_DIR)
    parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--mode", choices=["erase", "additive"], default="erase")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--parent-run", default=None)
    parser.add_argument("--no-manifest", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gt_root = repo_path(args.gt_root)
    splits_path = repo_path(args.splits)
    stage1_dir = repo_path(args.stage1_dir)
    model_dir = Path(repo_path(args.model_dir))
    out_dir = Path(repo_path(args.out_dir))
    splits = CVSplits.load(splits_path)
    run_id = args.run_id or artifact_id(out_dir)

    result = predict_oof(
        model_dir=model_dir, pred_out_dir=out_dir,
        stage1_dir=stage1_dir, splits=splits, gt_root=gt_root,
        dilation=args.dilation, mode=args.mode, prob_thresh=args.tau,
        gate=None, device=device)
    print(f"wrote {result['n_written']} merged predictions -> {out_dir}")

    if args.no_manifest:
        return 0

    manifest = prediction_manifest(
        result, model_dir, out_dir, stage1_dir, splits, gt_root,
        inference={"device": device, "mode": args.mode, "dilation": args.dilation, "tau": args.tau})
    manifest.run_id = run_id
    manifest.description = args.description or manifest.description
    manifest.tags = sorted(set(manifest.tags) | set(args.tag))
    manifest.parent_run = args.parent_run or manifest.parent_run
    manifest.provenance = provenance_now("scripts/predict_polar.py")
    destination = write_manifest(out_dir / "run.json", manifest)
    print(f"wrote manifest -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
