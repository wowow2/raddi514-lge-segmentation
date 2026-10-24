"""Launch the EMIDEC explorer.

Examples::

    python scripts/view.py                       # all cases, every run
    python scripts/view.py --split train --runs nnUNet_validation_pp
    python scripts/view.py --no-preds            # browse images + GT only
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="EMIDEC case explorer")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all",
                        help="which split to browse (default: all)")
    parser.add_argument("--no-preds", action="store_true",
                        help="browse without prediction overlays")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="run ids to load (default: every discovered run)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    import matplotlib
    matplotlib.use("Qt5Agg")
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    from src.viewer.app import MainWindow
    from src.viewer.model import ViewerModel
    from src.viewer.qt_theme import STYLE_SHEET

    model = ViewerModel(split=args.split, run_ids=args.runs,
                        use_predictions=not args.no_preds)
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)
    window = MainWindow(model)
    window.show()
    print(model.describe(), flush=True)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
