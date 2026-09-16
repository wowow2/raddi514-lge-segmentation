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

from src.datautils.emidec_dataset import EMIDECDataset
from src.viewer import DatasetViewer

SPLIT_DIRS = {
    "train": ["var/emidec_dataset/train"],
    "test": ["var/emidec_dataset/test"],
    "all": ["var/emidec_dataset/train", "var/emidec_dataset/test"],
}


def main():
    parser = argparse.ArgumentParser(description="EMIDEC dataset explorer")
    parser.add_argument(
        "--split", choices=["train", "test", "all"], default="all",
        help="which split to browse (default: all)",
    )
    args = parser.parse_args()

    dataset = EMIDECDataset(SPLIT_DIRS[args.split])
    print(f"Loaded {len(dataset)} cases [{_backend}] ({args.split}). Launching viewer...")
    viewer = DatasetViewer(dataset.cases)
    viewer.show()


if __name__ == "__main__":
    main()
