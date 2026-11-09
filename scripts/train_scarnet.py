"""CLI entry point for the ScarNet stage-2 trainer."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEFAULT_GT_ROOT,
    DEFAULT_SCARNET_MODEL_DIR,
    DEFAULT_SCARNET_PRED_DIR,
    DEFAULT_SPLITS_PATH,
)
from src.scarnet.checkpoint import import_class
from src.scarnet.loss import CombinedLoss
from src.scarnet.trainer import ScarNetTrainer, set_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a ScarNet stage-2 model")
    parser.add_argument("--fold", default="0",
                        help="Fold index or 'all' (default: 0)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-encoder", type=float, default=1e-5,
                        help="LR for trainable MedSAM encoder blocks; "
                             "models without param_groups ignore it")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-dir", default=DEFAULT_SCARNET_MODEL_DIR)
    parser.add_argument("--pred-dir", default=DEFAULT_SCARNET_PRED_DIR,
                        help="Stage-1 OOF predictions defining the ROI prior")
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--description", default=None)
    parser.add_argument("--no-metadata", action="store_true",
                        help="Skip run.json (useful for fold arrays)")

    parser.add_argument("--model-class", required=True,
                        help="'pkg.module:ClassName' for the network")
    parser.add_argument("--model-params", default=None,
                        help="JSON object passed to the model constructor; "
                             "use --out-size for the dataset, not out_size here")
    parser.add_argument("--model-name", default=None)

    parser.add_argument("--out-size", type=int, default=128)
    parser.add_argument("--pad", type=int, default=16)
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--multiple", type=int, default=16)
    parser.add_argument("--no-largest-component", action="store_true",
                        help="Use all stage-1 foreground for the ROI box")
    parser.add_argument("--clip-pct", type=float, nargs=2, default=(1.0, 99.0),
                        metavar=("LO", "HI"), help="Slice percentile clipping")

    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--select-metric",
                        choices=("lesion", "mean", "loss", "scar"),
                        default="lesion")
    parser.add_argument("--no-augment", action="store_true")

    parser.add_argument("--lambda1", type=float, default=0.5)
    parser.add_argument("--lambda2", type=float, default=0.4)
    parser.add_argument("--lambda3", type=float, default=0.1)
    parser.add_argument("--focal-alpha", type=float, default=0.7)
    parser.add_argument("--focal-beta", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-weights", default=None,
                        help="JSON list of CE weights (default: 0.1/0.3/0.3/0.3/0.3)")
    parser.add_argument("--no-adaptive-scar", action="store_true")
    parser.add_argument("--scar-boost", type=float, default=1.5)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    folds = list(range(5)) if args.fold.lower() == "all" else [int(args.fold)]
    model_params = json.loads(args.model_params) if args.model_params else {}
    out_size = args.out_size
    if "out_size" in model_params:
        model_out = int(model_params.pop("out_size"))
        if args.out_size == 128:
            out_size = model_out
            print(f"note: out_size={model_out} moved from --model-params "
                  "to the dataset flag")
        else:
            print(f"note: ignoring model_params.out_size={model_out}; "
                  f"using --out-size={args.out_size}")
    class_weights = (json.loads(args.class_weights)
                     if args.class_weights else None)
    model_cls = import_class(args.model_class)
    model_name = args.model_name or model_cls.__name__
    num_classes = int(model_params.get("num_classes", 5))

    print("=" * 70)
    print(f"SCARNET STAGE-2 | folds {folds} | epochs {args.epochs} | "
          f"batch {args.batch_size}")
    print(f"model {args.model_class} | params {model_params}")
    print(f"prior {args.pred_dir}")
    print("=" * 70)

    t_start = time.time()
    for fold in folds:
        print(f"\n>>> Training fold {fold}...")
        fold_seed = args.seed + fold
        set_seed(fold_seed)
        model = model_cls(**model_params)
        criterion = CombinedLoss(
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lambda3=args.lambda3,
            alpha=args.focal_alpha,
            beta=args.focal_beta,
            gamma=args.focal_gamma,
            class_weights=class_weights,
            adaptive_scar=not args.no_adaptive_scar,
            scar_boost=args.scar_boost,
            num_classes=num_classes,
        )
        trainer = ScarNetTrainer(
            fold=fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_encoder=args.lr_encoder,
            weight_decay=args.weight_decay,
            device=args.device,
            save_dir=args.save_dir,
            pred_dir=args.pred_dir,
            splits_path=args.splits,
            gt_root=args.gt_root,
            num_workers=args.num_workers,
            seed=fold_seed,
            resume=args.resume,
            model=model,
            model_class=args.model_class,
            model_params=model_params,
            criterion=criterion,
            out_size=out_size,
            pad=args.pad,
            min_size=args.min_size,
            multiple=args.multiple,
            clip_pct=tuple(args.clip_pct),
            use_largest_component=not args.no_largest_component,
            amp=not args.no_amp,
            grad_accum=args.grad_accum,
            select_metric=args.select_metric,
            augment=not args.no_augment,
            run_id=args.run_id,
            tags=args.tag,
            description=args.description,
            write_metadata=not args.no_metadata,
            model_name=model_name,
        )
        best = trainer.fit()
        print(f">>> fold {fold} done: {best}")

    print(f"\nFinished in {(time.time() - t_start) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
