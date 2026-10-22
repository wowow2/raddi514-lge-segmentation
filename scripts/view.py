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

from src.config import DEFAULT_GT_ROOT, repo_path
from src.datautils.emidec_dataset import EMIDECDataset
from src.runs import RunStore
from src.viewer import DatasetViewer

TEST_ROOT = str(repo_path("var/emidec_dataset/test"))
SPLIT_DIRS = {
    "train": [DEFAULT_GT_ROOT],
    "test": [TEST_ROOT],
    "all": [DEFAULT_GT_ROOT, TEST_ROOT],
}

def build_pred_index(cases, run_ids=None):
    """{case_id: {run_name: pred path or None}} from discovered runs."""
    store = RunStore()
    runs = store.select(run_ids) if run_ids else store.discover()
    usable = []
    for run in runs:
        index = run.prediction_index()
        if index is not None and len(index.available()):
            usable.append((run.name, index))
    pred_index = {}
    for case in cases:
        entry = {}
        for name, index in usable:
            if case.case_id in index:
                path = index.path(case.case_id)
                entry[name] = str(path) if path.exists() else None
            else:
                entry[name] = None
        pred_index[case.case_id] = entry
    return [name for name, _ in usable], pred_index


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
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="run ids to overlay (default: every discovered run with predictions)",
    )
    args = parser.parse_args()

    dataset = EMIDECDataset(SPLIT_DIRS[args.split])
    run_names, pred_index = (
        ([], {}) if args.no_preds else build_pred_index(dataset.cases, args.runs))
    print(f"Loaded {len(dataset)} cases [{_backend}] ({args.split})"
          + (f" + preds {run_names}." if run_names else " (no preds).")
          + " Launching viewer...")
    viewer = DatasetViewer(dataset.cases, run_names=run_names, pred_index=pred_index)
    viewer.show()


if __name__ == "__main__":
    main()
