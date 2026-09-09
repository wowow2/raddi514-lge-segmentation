"""Thin CLI wrapper around the nnUNet export module."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datautils import NnunetExporter
from src.datautils.emidec_dataset import EMIDECDataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export EMIDEC to nnUNet raw format")
    parser.add_argument("--out", default="nnUNet_raw", help="nnUNet_raw root directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train = EMIDECDataset("emidec_dataset/train")
    test = EMIDECDataset("emidec_dataset/test")
    task_dir = NnunetExporter(
        out_dir=args.out, overwrite=args.overwrite
    ).export(train.cases, test.cases)
    print(f"Exported {len(train)} train + {len(test)} test cases -> {task_dir}")


if __name__ == "__main__":
    main()
