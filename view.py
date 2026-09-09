import argparse

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

from src.emidec.emidec_dataset import EMIDECDataset
from src.emidec.viewer import DatasetViewer

SPLIT_DIRS = {
    "train": ["emidec_dataset/train"],
    "test": ["emidec_dataset/test"],
    "all": ["emidec_dataset/train", "emidec_dataset/test"],
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
