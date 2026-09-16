"""Thin CLI wrapper around the nnUNet export module."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datautils import (
    LABEL_NAMES,
    REGION_CLASS_ORDER,
    REGION_LABEL_NAMES,
    NnunetExporter,
)
from src.datautils.emidec_dataset import EMIDECDataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export EMIDEC to nnUNet raw format")
    parser.add_argument("--out", default="var/nnUNet_raw", help="nnUNet_raw root directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task-name", default="Dataset500_EMIDEC",
                        help="nnUNet task dir name (must be DatasetXXX_*)")
    parser.add_argument("--regions", action="store_true",
                        help="use BraTS-style overlapping region labels "
                             "(requires a distinct --task-name, e.g. Dataset501_*)")
    parser.add_argument("--neighbor-stack", action="store_true",
                        help="write 2.5D input channels (z-1/z/z+1, edge-replicated)")
    args = parser.parse_args()

    labels = dict(REGION_LABEL_NAMES) if args.regions else dict(LABEL_NAMES)
    order = tuple(REGION_CLASS_ORDER) if args.regions else None

    train = EMIDECDataset("var/emidec_dataset/train")
    test = EMIDECDataset("var/emidec_dataset/test")
    task_dir = NnunetExporter(
        out_dir=args.out, task_name=args.task_name, overwrite=args.overwrite,
        labels=labels, regions_class_order=order,
        neighbor_stack=args.neighbor_stack,
    ).export(train.cases, test.cases)
    print(f"Exported {len(train)} train + {len(test)} test cases -> {task_dir}")


if __name__ == "__main__":
    main()
