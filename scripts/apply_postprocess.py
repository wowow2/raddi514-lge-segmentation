"""Apply locked post-processing to a prediction directory, preserving layout.

Reads ``<pred-dir>/fold_*/validation/*.nii.gz``, writes the cleaned tree to
``--out-dir``, and records the ordered operations plus the parent run in a
``run.json`` manifest so the derivation is reproducible. Score the output with
``scripts/evaluate_nnunet_cv.py --pred-dir <out-dir>`` (or ``scripts/runs.py``).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_SPLITS_PATH
from src.datautils.postprocess import (
    MIN_MVO_VOXELS,
    MIN_SCAR_VOXELS,
    PostProcessor,
    apply_to_dir,
)
from src.datautils.predictions import CVPredictionIndex
from src.datautils.splits import CVSplits
from src.runs import RunManifest, provenance_now


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Post-process nnUNet predictions")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-scar-voxels", type=int, default=MIN_SCAR_VOXELS)
    parser.add_argument("--min-mvo-voxels", type=int, default=MIN_MVO_VOXELS)
    parser.add_argument("--scar-grow-px", type=int, default=0,
                        help="grow gated scar by N in-plane px into cavity/myo (0 = off)")
    parser.add_argument("--run-id", default=None,
                        help="run id (default: output directory name)")
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--parent-run", default=None,
                        help="run id that produced --pred-dir (default: its name)")
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--expected-outputs", choices=["unknown", "oof", "all_cases"],
                        default="unknown",
                        help="whether predictions cover OOF val cases or all cases")
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
    out_dir = Path(args.out_dir)
    post = PostProcessor(min_scar_voxels=args.min_scar_voxels,
                         min_mvo_voxels=args.min_mvo_voxels,
                         scar_grow_px=args.scar_grow_px)
    print(f"using {post}")

    written = apply_to_dir(pred_dir, out_dir, post)
    print(f"post-processed {len(written)} files -> {out_dir}")

    if args.no_manifest:
        return 0

    run_id = args.run_id or out_dir.name
    parent = _existing_manifest(pred_dir)
    splits = None
    try:
        splits = CVSplits.load(args.splits, validate=False)
    except (OSError, ValueError, json.JSONDecodeError):
        pass

    manifest = RunManifest.from_dict(parent)
    manifest.run_id = run_id
    manifest.description = args.description or manifest.description
    manifest.display_name = args.display_name or manifest.display_name
    manifest.tags = sorted(set(manifest.tags) | set(args.tag))
    manifest.parent_run = args.parent_run or manifest.parent_run or pred_dir.name
    manifest.postprocess = {
        "ops": post.ops(),
        "min_scar_voxels": post.min_scar_voxels,
        "min_mvo_voxels": post.min_mvo_voxels,
        "scar_grow_px": post.scar_grow_px,
    }
    manifest.paths.update({"predictions": str(out_dir.resolve()), "metrics": None})
    if splits is not None:
        manifest.data.update({
            "splits_path": args.splits,
            "splits_sha256": splits.sha256,
            "n_folds": len(splits.folds),
            "n_cases": len(splits.case_ids()),
        })
    manifest.status["expected_outputs"] = (
        None if args.expected_outputs == "unknown" else args.expected_outputs)
    if splits is not None:
        index = CVPredictionIndex(out_dir, splits)
        manifest.status["prediction"] = index.summary()
    manifest.provenance = provenance_now("scripts/apply_postprocess.py")

    path = out_dir / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    print(f"wrote manifest -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
