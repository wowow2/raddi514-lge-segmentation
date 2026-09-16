import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib


def _ensure_interactive_backend() -> str:
    """Select a GUI backend so plt.show() actually opens a window.

    Without this, environments defaulting to Agg silently show nothing.
    """
    for backend in ("QtAgg", "TkAgg"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt

            plt.figure()  # validates the backend plugin loads
            plt.close("all")
            return backend
        except Exception:
            continue
    raise RuntimeError(
        "No interactive matplotlib backend available (tried QtAgg, TkAgg). "
        "Install one, e.g.: .venv/bin/pip install PyQt5"
    )


_backend = _ensure_interactive_backend()

from src.config import case_id_to_nnunet_id
from src.datautils.emidec_dataset import EMIDECDataset
from src.viewer import DatasetViewer

SPLIT_DIRS = {
    "train": ["var/emidec_dataset/train"],
    "test": ["var/emidec_dataset/test"],
    "all": ["var/emidec_dataset/train", "var/emidec_dataset/test"],
}

# (overlay name, post-processed predictions dir) in cycle order.
PRED_RUNS = [
    ("plain", "var/nnUNet_validation_pp"),
    ("resenc", "var/nnUNet_validation_500resenc_pp"),
    ("regions", "var/nnUNet_validation_501_pp"),
]
SPLITS_PATH = "var/nnUNet_validation/splits_final.json"


def build_pred_index(cases):
    """{case_id: {run_name: pred path or None}}; missing dirs degrade to {}."""
    import json

    with open(SPLITS_PATH) as f:
        folds = json.load(f)
    fold_of = {nid: fi for fi, s in enumerate(folds) for nid in s["val"]}
    runs = [(n, d) for n, d in PRED_RUNS if os.path.isdir(d)]
    index = {}
    for case in cases:
        nid = case_id_to_nnunet_id(case.case_id)
        fi = fold_of.get(nid)
        entry = {}
        for name, d in runs:
            p = os.path.join(d, f"fold_{fi}", "validation", f"{nid}.nii.gz")
            entry[name] = p if fi is not None and os.path.exists(p) else None
        index[case.case_id] = entry
    return [n for n, _ in runs], index


def main():
    parser = argparse.ArgumentParser(description="EMIDEC dataset explorer")
    parser.add_argument(
        "--split", choices=["train", "test", "all"], default="all",
        help="which split to browse (default: all)",
    )
    parser.add_argument(
        "--no-preds", action="store_true",
        help="browse without prediction overlays",
    )
    args = parser.parse_args()

    dataset = EMIDECDataset(SPLIT_DIRS[args.split])
    run_names, pred_index = ([], {}) if args.no_preds else build_pred_index(dataset.cases)
    print(f"Loaded {len(dataset)} cases [{_backend}] ({args.split})"
          + (f" + preds {run_names}." if run_names else " (no preds).")
          + " Launching viewer...")
    viewer = DatasetViewer(dataset.cases, run_names=run_names, pred_index=pred_index)
    viewer.show()


if __name__ == "__main__":
    main()
