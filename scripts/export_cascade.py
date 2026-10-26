"""Export a 3D ROI cascade dataset from OOF 2D stage-1 predictions.

The ROI always comes from stage-1; ``--no-probs`` only omits the probability
channels from the input. The same 5-fold splits are copied beside the dataset.

Examples::

    python scripts/export_cascade.py --limit 5 --out /tmp/cascade_smoke
    python scripts/export_cascade.py --overwrite
    python scripts/export_cascade.py --no-probs --task-name Dataset505_EMIDEC_ROI
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cascade import ExportConfig, export_dataset
from src.config import (DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH, DEFAULT_NNUNET_RAW, artifact_default)

DEFAULT_TASK = "Dataset503_EMIDEC_Cascade"
DEFAULT_PROB_DIR = (
    "nnunet://nnUNet_results/Dataset500_EMIDEC/"
    "probs_nnUNetResEncUNetMPlans_2d_"
    "nnUNetTrainer_probabilisticOversampling"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export ROI 3D cascade dataset")
    parser.add_argument("--out", default=DEFAULT_NNUNET_RAW)
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--prob-dir", default=artifact_default(
        "nnUNet_validation_500over", "probabilities", DEFAULT_PROB_DIR))
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    parser.add_argument("--pad", type=int, default=16,
                        help="in-plane padding in voxels (default 16)")
    parser.add_argument("--min-size", type=int, default=64,
                        help="minimum crop side (default 64)")
    parser.add_argument("--multiple", type=int, default=16,
                        help="round crop side up to this multiple (default 16)")
    parser.add_argument("--no-probs", action="store_true",
                        help="omit stage-1 probability channels; the ROI still "
                             "uses stage-1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="export only the first N cases (smoke test)")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="specific case ids, e.g. Case_N025 Case_P064")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = export_dataset(ExportConfig(
        out_dir=args.out,
        task_name=args.task_name,
        prob_dir=args.prob_dir,
        splits_path=args.splits,
        gt_root=args.gt_root,
        pad=args.pad,
        min_size=args.min_size,
        multiple=args.multiple,
        include_probs=not args.no_probs,
        overwrite=args.overwrite,
        limit=args.limit,
        case_ids=tuple(args.cases) if args.cases else None,
    ))

    sizes = Counter(tuple(case.crop_shape[:2]) for case in summary.cases)
    print(f"wrote {summary.n_cases} cases -> "
          f"{Path(args.out) / summary.task_name}")
    print(f"channels: {len(summary.channels)} | "
          f"crops: {', '.join(f'{w}x{h}' for w, h in sorted(sizes))}")
    for case in summary.cases[:5]:
        print(f"  {case.case_id}: bbox {case.bbox} crop {case.crop_shape}")
    if summary.n_cases > 5:
        print(f"  ... {summary.n_cases - 5} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
