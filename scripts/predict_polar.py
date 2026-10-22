"""Run polar out-of-fold inference/merge and save ungated labels.

This deliberately does not post-process or score. Downstream::

    python scripts/apply_postprocess.py --pred-dir var/polar_preds_raw \
        --out-dir var/polar_preds_pp --min-scar-voxels 50 --scar-grow-px 0
    python scripts/evaluate_nnunet_cv.py --pred-dir var/polar_preds_pp \
        --out var/results/polar_pp_metrics.json

Changing gates or scar growth then never requires another forward pass.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH
from src.datautils.splits import CVSplits
from src.polar.predict import predict_oof
from src.runs import RunManifest, provenance_now


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Polar cascade prediction (no scoring)")
    parser.add_argument("--model-dir", default="var/polar_models")
    parser.add_argument("--out-dir", default="var/polar_preds_raw")
    parser.add_argument("--stage1-dir", default="var/nnUNet_validation_501_pp")
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
    splits = CVSplits.load(args.splits)
    out_dir = Path(args.out_dir)
    run_id = args.run_id or out_dir.name

    result = predict_oof(
        model_dir=args.model_dir, pred_out_dir=out_dir,
        stage1_dir=args.stage1_dir, splits=splits, gt_root=args.gt_root,
        dilation=args.dilation, mode=args.mode, prob_thresh=args.tau,
        gate=None, device=device)
    print(f"wrote {result['n_written']} merged predictions -> {out_dir}")

    if args.no_manifest:
        return 0

    parent_path = Path(args.model_dir) / "run.json"
    parent = {}
    if parent_path.exists():
        with open(parent_path) as f:
            parent = json.load(f)
    parent_manifest = RunManifest.from_dict(parent)
    config = result["config"]
    manifest = RunManifest(
        run_id=run_id,
        description=args.description or parent_manifest.description,
        tags=sorted(set(parent_manifest.tags) | set(args.tag)),
        stage="stage2",
        parent_run=args.parent_run or parent_manifest.run_id or Path(args.model_dir).name,
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
        postprocess={"ops": []},
        paths={"predictions": str(out_dir.resolve()),
               "checkpoints": str(Path(args.model_dir).resolve())},
        status={"expected_outputs": "oof",
                "prediction": result["prediction"]},
        provenance=provenance_now("scripts/predict_polar.py"),
    )
    with open(out_dir / "run.json", "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    print(f"wrote manifest -> {out_dir / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
