"""Apply locked post-processing to a nnUNet predictions dir, preserving layout.

Reads <pred-dir>/fold_*/validation/*.nii.gz, writes the cleaned tree to
--out-dir. Then score with scripts/evaluate_nnunet_cv.py --pred-dir <out-dir>.
"""

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datautils.postprocess import MIN_MVO_VOXELS, MIN_SCAR_VOXELS, PostProcessor


def main():
    parser = argparse.ArgumentParser(description="Post-process nnUNet predictions")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-scar-voxels", type=int, default=MIN_SCAR_VOXELS)
    parser.add_argument("--min-mvo-voxels", type=int, default=MIN_MVO_VOXELS)
    args = parser.parse_args()
    post = PostProcessor(min_scar_voxels=args.min_scar_voxels,
                         min_mvo_voxels=args.min_mvo_voxels)
    print(f"using {post}")

    n = 0
    for fold in sorted(Path(args.pred_dir).glob("fold_*")):
        for src in sorted((fold / "validation").glob("*.nii.gz")):
            dst = Path(args.out_dir) / fold.name / "validation" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = nib.load(str(src))
            cleaned = post(np.asanyarray(img.dataobj).astype(np.uint8))
            nib.save(nib.Nifti1Image(cleaned, img.affine, img.header), str(dst))
            n += 1
    print(f"post-processed {n} files -> {args.out_dir}")


if __name__ == "__main__":
    main()
