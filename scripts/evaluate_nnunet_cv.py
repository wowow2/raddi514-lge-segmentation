"""Score nnUNet cross-validation predictions against EMIDEC ground truth.

Writes a schema-v2 metrics JSON (tidy per-case/per-class rows + manifest) and,
unless ``--no-manifest`` is given, records ``run.json`` in the prediction
directory so the run is discoverable by ``scripts/runs.py``.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEFAULT_GT_ROOT,
    DEFAULT_OUT_PATH,
    DEFAULT_PRED_DIR,
    DEFAULT_SPLITS_PATH,
)
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.evaluator import DatasetEvaluator
from src.datautils.predictions import CVPredictionIndex
from src.datautils.splits import CVSplits
from src.runs import RunManifest, provenance_now


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Score nnUNet CV Predictions")
    parser.add_argument("--pred-dir", default=DEFAULT_PRED_DIR)
    parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--run-id", default=None,
                        help="run id (default: prediction directory name)")
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--parent-run", default=None)
    parser.add_argument("--model", default=None, help="model name for the manifest")
    parser.add_argument("--trainer", default=None, help="nnU-Net trainer class")
    parser.add_argument("--plans", default=None, help="nnU-Net plans identifier")
    parser.add_argument("--config-name", default=None,
                        help="nnU-Net configuration, e.g. 2d")
    parser.add_argument("--label-scheme", default=None,
                        help="label scheme, e.g. flat or regions")
    parser.add_argument("--channels", type=int, default=None,
                        help="number of input channels")
    parser.add_argument("--checkpoints", default=None,
                        help="training checkpoint directory to record")
    parser.add_argument("--checkpoint", choices=["best", "final"], default=None,
                        help="checkpoint used for prediction, when known")
    parser.add_argument("--probabilities", default=None,
                        help="probability directory to record")
    parser.add_argument("--allow-partial", action="store_true",
                        help="score existing files and report the rest as missing")
    parser.add_argument("--no-manifest", action="store_true")
    return parser.parse_args(argv)


def _existing_manifest(pred_dir: Path) -> dict:
    path = pred_dir / "run.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def main(argv=None) -> int:
    args = parse_args(argv)
    pred_dir = Path(args.pred_dir)

    dataset = EMIDECDataset(args.gt_root)
    splits = CVSplits.load(args.splits)
    validation = splits.validate(case_ids=dataset.case_ids)
    validation.raise_if_invalid()
    for warning in validation.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    index = CVPredictionIndex(pred_dir, splits)
    missing = index.missing()
    if missing and not args.allow_partial:
        raise FileNotFoundError(
            f"{len(missing)} prediction file(s) missing under {pred_dir} "
            f"(first: {missing[0]}). Use --allow-partial to score what exists.")
    if missing:
        print(f"warning: scoring partial run; {len(missing)} case(s) missing",
              file=sys.stderr)

    evaluator = DatasetEvaluator()
    for case_id in index.available():
        fold = index.fold_of(case_id)
        evaluator.add_emidec_case(
            case=dataset.get(case_id), pred=index.load(case_id),
            metadata={"fold": fold})

    parent = _existing_manifest(pred_dir)
    run_id = args.run_id or parent.get("run_id") or pred_dir.name
    report = evaluator.aggregate(run_id=run_id)
    manifest = RunManifest.from_dict(parent)
    manifest.run_id = run_id
    manifest.description = args.description or manifest.description
    manifest.display_name = args.display_name or manifest.display_name
    manifest.tags = sorted(set(manifest.tags) | set(args.tag))
    manifest.parent_run = args.parent_run or manifest.parent_run
    manifest.stage = manifest.stage or "stage1"
    if args.model:
        manifest.model["name"] = args.model
    for key, value in (("trainer", args.trainer), ("plans", args.plans),
                       ("configuration", args.config_name)):
        if value:
            manifest.model[key] = value
    if args.label_scheme:
        manifest.data["label_scheme"] = args.label_scheme
    if args.channels is not None:
        manifest.data["channels"] = args.channels
    manifest.data.update({
        "gt_root": args.gt_root,
        "splits_path": args.splits,
        "splits_sha256": splits.sha256,
        "n_folds": len(splits.folds),
        "n_cases": len(splits.case_ids()),
    })
    manifest.paths.update({
        "predictions": str(pred_dir.resolve()),
        "metrics": str(Path(args.out).resolve()),
    })
    if args.checkpoints:
        manifest.paths["checkpoints"] = str(Path(args.checkpoints).resolve())
    if args.checkpoint:
        manifest.inference["checkpoint"] = args.checkpoint
    if args.probabilities:
        manifest.paths["probabilities"] = str(Path(args.probabilities).resolve())
    manifest.status["prediction"] = index.summary()
    manifest.status["missing_cases"] = missing
    manifest.status["expected_outputs"] = "oof"
    manifest.provenance = provenance_now("scripts/evaluate_nnunet_cv.py")
    report.manifest = manifest.to_dict()

    report.save_json(args.out)
    if not args.no_manifest:
        path = pred_dir / "run.json"
        with open(path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    print("\n" + report.to_markdown())
    print(f"\nSaved metrics -> {args.out} ({report.overall['n_cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
