"""Score nnUNet cross-validation predictions against EMIDEC ground truth."""

import argparse
import json
import os
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEFAULT_GT_ROOT,
    DEFAULT_OUT_PATH,
    DEFAULT_PRED_DIR,
    DEFAULT_SPLITS_PATH,
    nnunet_id_to_case_id,
)
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.evaluator import DatasetEvaluator


def main():
    parser = argparse.ArgumentParser(description="Score nnUNet CV Predictions")
    parser.add_argument("--pred-dir", default=DEFAULT_PRED_DIR, help="Path to nnUNet_validation")
    parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT, help="Path to emidec_dataset/train")
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH, help="Path to splits_final.json")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Path to output results JSON")
    args = parser.parse_args()

    dataset = EMIDECDataset(args.gt_root)
    evaluator = DatasetEvaluator()

    with open(args.splits) as f:
        folds = json.load(f)

    for fold_idx, split in enumerate(folds):
        for nnunet_id in split["val"]:
            case_id = nnunet_id_to_case_id(nnunet_id)
            val_folder = os.path.join(args.pred_dir, f"fold_{fold_idx}", "validation")
            pred_file = os.path.join(val_folder, f"{nnunet_id}.nii.gz")

            if not os.path.exists(pred_file):
                raise FileNotFoundError(f"Missing prediction: {pred_file}")

            pred = nib.load(pred_file).get_fdata().astype(np.uint8)

            case = dataset.get(case_id)
            evaluator.add_emidec_case(case=case, pred=pred, metadata={"fold": fold_idx})

    report = evaluator.aggregate()
    report.save_json(args.out)

    print("\n" + report.to_markdown())
    print(f"\nSaved metrics -> {args.out} ({report.overall['n_cases']} cases)")


if __name__ == "__main__":
    main()
