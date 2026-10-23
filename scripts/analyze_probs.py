"""Scar-vs-cavity probability diagnostic + scar-threshold sweep (out-of-fold).

Usage (local, partial runs are fine; missing cases are skipped):
    python scripts/analyze_probs.py \
        --prob-dir var/nnUNet_results/Dataset500_EMIDEC/probs_nnUNetResEncUNetMPlans_2d

Writes a small JSON (var/results/prob_diag.json) with per-case diagnostics and
the threshold sweep. Sweep scoring uses the repo's Dice convention
(both-empty -> 1.0) and the locked 50-voxel scar gate.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.probs import (
    DEFAULT_THRESHOLDS,
    ProbIndex,
    predict_with_scar_threshold,
    scar_diagnostics,
)
from src.config import DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH, LABEL_SCAR, repo_path
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.metrics import SegmentationMetrics
from src.datautils.postprocess import PostProcessor

DEFAULT_PROB_DIR = "var/nnUNet_results/Dataset500_EMIDEC/probs_nnUNetResEncUNetMPlans_2d"
HARD_CASES = ["Case_P064", "Case_P088", "Case_P011", "Case_P066", "Case_P051", "Case_P003"]


def parse_args():
    p = argparse.ArgumentParser(description="Scar-vs-cavity probability diagnostic")
    p.add_argument("--prob-dir", default=DEFAULT_PROB_DIR)
    p.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    p.add_argument("--splits", default=DEFAULT_SPLITS_PATH)
    p.add_argument("--out", default="var/results/prob_diag.json")
    p.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    p.add_argument("--hard", nargs="*", default=HARD_CASES, help="case ids to print in detail")
    p.add_argument("--min-verify", type=float, default=0.999,
                   help="abort if argmax(prob) vs exported .nii.gz match is below this")
    return p.parse_args()


def fmt(x, nd=3):
    return "  -  " if x is None else f"{x:.{nd}f}"


def main():
    args = parse_args()
    thresholds = tuple(args.thresholds)
    dataset = EMIDECDataset(repo_path(args.gt_root))
    index = ProbIndex(args.prob_dir, args.splits)
    post = PostProcessor()
    scar_dice = SegmentationMetrics((LABEL_SCAR,)).dice

    cases = index.available()
    if not cases:
        raise SystemExit(f"No .npz found under {args.prob_dir}")
    print(f"{len(cases)} cases with probabilities (of {len(index.paths)} in the splits)")

    diag: dict[str, dict] = {}
    # sweep[key][case_id] = scar Dice after the 50-voxel gate; key "argmax" or "t=<x>"
    sweep: dict[str, dict[str, float]] = {"argmax": {}, **{f"t={t}": {} for t in thresholds}}
    groups: dict[str, str] = {}
    worst_match = 1.0

    for cid in cases:
        case = dataset.get(cid)
        _, gt = case.load()
        prob = index.load(cid)
        if prob.shape[1:] != gt.shape:
            raise ValueError(f"{cid}: prob {prob.shape[1:]} vs gt {gt.shape}")

        match = index.verify(cid, prob)
        worst_match = min(worst_match, match)
        if match < args.min_verify:
            raise SystemExit(f"{cid}: argmax matches exported labels only {match:.4f}; "
                             "axis order or file pairing is wrong")

        groups[cid] = case.group
        diag[cid] = {"group": case.group, **scar_diagnostics(prob, gt, thresholds)}

        for key, t in [("argmax", None)] + [(f"t={t}", t) for t in thresholds]:
            pred = post.gate_small_scar(predict_with_scar_threshold(prob, t))
            sweep[key][cid] = scar_dice(pred, gt)[LABEL_SCAR]

    print(f"verify: worst argmax-vs-label match = {worst_match:.5f}\n")

    # ---- sweep summary -----------------------------------------------------
    path_ids = [c for c in cases if groups[c] == "Pathological"]
    norm_ids = [c for c in cases if groups[c] == "Normal"]
    pos_ids = [c for c in path_ids if diag[c]["gt_scar"] > 0]
    print(f"cases: {len(path_ids)} pathological ({len(pos_ids)} with scar GT), {len(norm_ids)} normal")
    print("\nScar Dice after the 50-voxel gate")
    print(f"{'rule':<10} {'all':>6} {'scar-pos':>9} {'normal FPs':>11}")
    summary = {}
    for key, per in sweep.items():
        allm = float(np.mean([per[c] for c in cases]))
        posm = float(np.mean([per[c] for c in pos_ids])) if pos_ids else None
        fps = sum(1 for c in norm_ids if per[c] < 1.0)
        summary[key] = {"all": allm, "scar_present": posm, "normal_fps": fps}
        print(f"{key:<10} {allm:>6.3f} {fmt(posm):>9} {fps:>6}/{len(norm_ids)}")

    # ---- hard cases + cohort medians --------------------------------------
    cols = ["gt_scar", "frac_gt_scar_pred_cav", "mean_p_scar_in_lost",
            "mean_p_cav_in_lost", f"recall_at_{thresholds[0]}", "recall_at_0.3"]
    keys_seen = set().union(*(diag[c].keys() for c in pos_ids)) if pos_ids else set()
    cols = [c for c in cols if c in keys_seen]
    print("\nHard cases (P(scar)/P(cav) are means over GT-scar voxels the argmax gave to cavity)")
    print(f"{'case':<10}" + "".join(f"{c[:22]:>24}" for c in cols))
    for cid in args.hard:
        d = diag.get(cid)
        if d is None:
            print(f"{cid:<10} (no probabilities yet)")
            continue
        print(f"{cid:<10}" + "".join(f"{fmt(d.get(c)):>24}" for c in cols))
    print(f"{'median':<10}" + "".join(
        f"{fmt(float(np.median([diag[c_][col] for c_ in pos_ids if col in diag[c_]])) if pos_ids else None):>24}"
        for col in cols))

    # ---- save --------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_path = repo_path(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"prob_dir": args.prob_dir, "thresholds": thresholds,
                   "summary": summary, "sweep": sweep, "per_case": diag}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()