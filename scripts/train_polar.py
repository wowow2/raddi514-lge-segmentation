"""CLI entry point to train Polar nnU-Net across cross-validation folds."""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_POLAR_MODEL_DIR, DEFAULT_STAGE1_PRED_DIR, DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH
from src.polar.trainer import PolarTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Polar nnU-Net on EMIDEC")
    parser.add_argument(
        "--fold",
        default="0",
        help="Fold index (0, 1, 2, 3, 4) or 'all' to train all 5 folds (default: 0)",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--use-radius", action="store_true", help="Add CoordConv normalized radius channel")
    parser.add_argument(
        "--no-deep-supervision",
        action="store_true",
        help="Disable auxiliary deep supervision heads",
    )
    parser.add_argument("--device", default=None, help="Device to use ('cuda', 'cpu', or auto-detect)")
    parser.add_argument(
        "--save-dir",
        default=DEFAULT_POLAR_MODEL_DIR,
        help=f"Root directory for saving checkpoints (default: {DEFAULT_POLAR_MODEL_DIR})",
    )
    parser.add_argument(
        "--pred-dir",
        default=DEFAULT_STAGE1_PRED_DIR,
        help=f"Path to Stage 1 predictions for anchors (default: {DEFAULT_STAGE1_PRED_DIR})",
    )
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH, help="Path to splits_final.json")
    parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT, help="Path to emidec_dataset/train")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint_last.pt in each fold directory")
    parser.add_argument("--run-id", default=None, help="Run id stored in run.json")
    parser.add_argument("--tag", action="append", default=[], help="Run tag (repeatable)")
    parser.add_argument("--description", default=None)
    parser.add_argument("--model-name", default="polar_unet",
                        help="Model name recorded in metadata (default: polar_unet)")
    parser.add_argument("--model-params", default=None,
                        help="JSON object merged into the model constructor params")
    parser.add_argument("--model-class", default=None,
                        help="optional 'package.module:ClassName' to construct instead "
                             "of PolarUNet (the dataset still supplies one input channel)")
    return parser.parse_args()


def _load_class(spec: str):
    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise SystemExit(f"--model-class must look like 'pkg.module:Class', got {spec!r}")
    return getattr(importlib.import_module(module_name), class_name)


def main():
    args = parse_args()
    folds = range(5) if args.fold.lower() == "all" else [int(args.fold)]

    print("=" * 70)
    print(f"POLAR nnU-NET TRAINING | Folds: {list(folds)} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"Use Radius Channel: {args.use_radius} | Deep Supervision: {not args.no_deep_supervision}")
    print(f"Seed: {args.seed} | Resume: {args.resume} | Run id: {args.run_id or args.save_dir}")
    print("=" * 70)

    model_params = json.loads(args.model_params) if args.model_params else {}
    model_cls = _load_class(args.model_class) if args.model_class else None
    model_name = model_cls.__name__ if model_cls is not None else args.model_name

    t_start = time.time()
    for f in folds:
        print(f"\n>>> Launching Training for Fold {f}...")
        fold_model = model_cls(**model_params) if model_cls is not None else None
        trainer = PolarTrainer(
            fold=f,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_radius=args.use_radius,
            deep_supervision=not args.no_deep_supervision,
            device=args.device,
            save_dir=args.save_dir,
            pred_dir=args.pred_dir,
            splits_path=args.splits,
            gt_root=args.gt_root,
            num_workers=args.num_workers,
            seed=args.seed,
            resume=args.resume,
            run_id=args.run_id,
            tags=args.tag,
            description=args.description,
            model_name=model_name,
            model_params=model_params or None,
            model=fold_model,
        )
        best_ckpt = trainer.fit()
        print(f">>> Fold {f} complete! Best model: {best_ckpt}")

    elapsed = (time.time() - t_start) / 60.0
    print(f"\nAll requested folds completed in {elapsed:.1f} minutes.")


if __name__ == "__main__":
    main()
