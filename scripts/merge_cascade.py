"""Merge ROI-cascade OOF predictions back into full-volume label space.

The Dataset503 stage-2 predictions live in square in-plane ROI crop space;
their per-case bbox is recorded in ``cascade_manifest.json`` at export.
This step pastes each prediction into a zeroed full-volume label map so the
regular gated postprocessing and evaluator (full GT geometry) can run.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cascade.merge import merge_oof


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Merge ROI Cascade Predictions")
    parser.add_argument("--pred-dir", required=True,
                        help="directory with fold_*/validation ROI predictions")
    parser.add_argument("--manifest", required=True,
                        help="cascade_manifest.json from the Dataset503 export")
    parser.add_argument("--gt-root", required=True,
                        help="EMIDEC train root (Case_*/Images reference geometry)")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results = merge_oof(args.pred_dir, args.manifest, args.gt_root, args.out_dir)
    print(f"merged {len(results)} cases -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
