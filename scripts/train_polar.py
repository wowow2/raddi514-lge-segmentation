"""CLI entry point to train Polar nnU-Net across cross-validation folds."""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH
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
        default="var/polar_models",
        help="Root directory for saving checkpoints (default: var/polar_models)",
    )
    parser.add_argument(
        "--pred-dir",
        default="var/nnUNet_validation_501_pp",
        help="Path to Stage 1 predictions for anchors (default: var/nnUNet_validation_501_pp)",
    )
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH, help="Path to splits_final.json")
    parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT, help="Path to emidec_dataset/train")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default: 2)")
    return parser.parse_args()


def main():
    args = parse_args()
    folds = range(5) if args.fold.lower() == "all" else [int(args.fold)]

    print("=" * 70)
    print(f"POLAR nnU-NET TRAINING | Folds: {list(folds)} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"Use Radius Channel: {args.use_radius} | Deep Supervision: {not args.no_deep_supervision}")
    print("=" * 70)

    t_start = time.time()
    for f in folds:
        print(f"\n>>> Launching Training for Fold {f}...")
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
        )
        best_ckpt = trainer.fit()
        print(f">>> Fold {f} complete! Best model: {best_ckpt}")

    elapsed = (time.time() - t_start) / 60.0
    print(f"\nAll requested folds completed in {elapsed:.1f} minutes.")


if __name__ == "__main__":
    main()