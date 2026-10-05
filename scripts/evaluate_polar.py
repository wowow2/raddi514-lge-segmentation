"""Evaluate Polar Cascade out-of-fold predictions against ground truth."""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEFAULT_GT_ROOT,
    DEFAULT_SPLITS_PATH,
    case_id_to_nnunet_id,
    nnunet_id_to_case_id,
)
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.evaluator import DatasetEvaluator
from src.datautils.postprocess import MIN_SCAR_VOXELS, PostProcessor
from src.polar.anchors import AnchorIndex
from src.polar.merge import PolarMerger
from src.polar.model import PolarUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Polar Cascade Cross-Validation")
    parser.add_argument(
        "--model-dir",
        default="var/polar_models",
        help="Root directory holding fold_* checkpoints (default: var/polar_models)",
    )
    parser.add_argument(
        "--pred-out-dir",
        default="var/polar_preds",
        help="Output directory to save final merged NIfTI predictions (default: var/polar_preds)",
    )
    parser.add_argument(
        "--stage1-dir",
        default="var/nnUNet_validation_501_pp",
        help="Stage 1 predictions directory (default: var/nnUNet_validation_501_pp)",
    )
    parser.add_argument(
        "--out",
        default="var/results/polar_cv_metrics.json",
        help="Path to save evaluation metrics JSON (default: var/results/polar_cv_metrics.json)",
    )
    parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT, help="Path to emidec_dataset/train")
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH, help="Path to splits_final.json")
    parser.add_argument("--dilation", type=int, default=2, help="In-plane paint-over dilation (default: 2)")
    parser.add_argument("--min-scar-voxels", type=int, default=MIN_SCAR_VOXELS, help="Post-processor scar gate threshold")
    parser.add_argument("--no-gate", action="store_true", help="Disable the 50-voxel post-processor scar gate")
    parser.add_argument("--device", default=None, help="Device to use ('cuda' or 'cpu')")
    return parser.parse_args()


def load_fold_model(ckpt_path: Path, device: str) -> PolarUNet:
    """Load PolarUNet weights and flags from checkpoint dict."""
    ckpt = torch.load(str(ckpt_path), map_location=device)
    use_radius = ckpt.get("use_radius", False)
    deep_sup = ckpt.get("deep_supervision", True)

    model = PolarUNet(
        in_channels=1,
        num_classes=5,
        use_radius=use_radius,
        deep_supervision=deep_sup,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gate = None if args.no_gate else PostProcessor(min_scar_voxels=args.min_scar_voxels)

    dataset = EMIDECDataset(args.gt_root)
    evaluator = DatasetEvaluator()
    anchors_index = AnchorIndex(pred_dir=args.stage1_dir, splits_path=args.splits)

    with open(args.splits) as f:
        splits = json.load(f)

    print("=" * 70)
    print("POLAR CASCADE OUT-OF-FOLD EVALUATION")
    print(f"Models: {args.model_dir} | Dilation: {args.dilation} | Gate: {'Disabled' if args.no_gate else args.min_scar_voxels}")
    print(f"Output: {args.pred_out_dir}")
    print("=" * 70)

    total_evaluated = 0

    for fold_idx, split in enumerate(splits):
        ckpt_path = Path(args.model_dir) / f"fold_{fold_idx}" / "checkpoint_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold_idx}: {ckpt_path}")

        print(f"\n--- Evaluating Fold {fold_idx} ({len(split['val'])} cases) ---")
        model = load_fold_model(ckpt_path, device=device)
        merger = PolarMerger(model=model, dilation=args.dilation, gate=gate, device=device)

        fold_val_dir = Path(args.pred_out_dir) / f"fold_{fold_idx}" / "validation"
        fold_val_dir.mkdir(parents=True, exist_ok=True)

        for nnunet_id in split["val"]:
            cid = nnunet_id_to_case_id(nnunet_id)
            case = dataset.get(cid)
            raw_img, _ = case.load()

            case_anchors = anchors_index.get(cid)
            nnunet_pred = case_anchors.pred_vol

            # Run PolarMerger (Cartesian warp + paint-over rule + 50-voxel gate)
            merged_pred = merger(
                raw_vol=raw_img,
                nnunet_pred=nnunet_pred,
                case_anchors=case_anchors,
                apply_gate=(gate is not None),
            )

            # Save NIfTI preserving exact header and affine
            nii_ref = nib.load(str(case_anchors.pred_path))
            out_nii_path = fold_val_dir / f"{nnunet_id}.nii.gz"
            nib.save(
                nib.Nifti1Image(merged_pred.astype(np.uint8), nii_ref.affine, nii_ref.header),
                str(out_nii_path),
            )

            # Evaluate case against ground truth
            evaluator.add_emidec_case(case=case, pred=merged_pred, metadata={"fold": fold_idx})
            total_evaluated += 1

    report = evaluator.aggregate()
    report.save_json(args.out)

    print("\n" + "=" * 50)
    print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 50)
    print(report.to_markdown())
    print(f"\nSaved metrics -> {args.out} ({report.overall['n_cases']} cases)")


if __name__ == "__main__":
    main()