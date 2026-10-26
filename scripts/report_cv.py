"""CV reports: --html for all experiments, or plots and a bad-performer table.

By default this reads saved ``metrics`` files (fast, no volumes). Use
``--images`` to compute the worst-slice column from prediction NIfTIs, or
``--from-images`` to ignore metrics entirely.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH, DEFAULT_REPORT_DIR, REPO_ROOT, repo_path
from src.results import ResultsTable
from src.runs import RunStore


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CV HTML reporting, plots and tables")
    parser.add_argument("--html", nargs="?", const=str(Path(DEFAULT_REPORT_DIR) / "experiments.html"),
                        help="generate a standalone report of all canonical outputs; optionally give its path")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="run ids (default: every discovered run)")
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--case-set", choices=["intersection", "union"],
                        default="intersection")
    parser.add_argument("--images", action="store_true",
                        help="load prediction volumes for the worst-slice column")
    parser.add_argument("--from-images", action="store_true",
                        help="compute Dice from NIfTIs instead of saved metrics")
    parser.add_argument("--no-bad-table", action="store_true")
    return parser.parse_args(argv)


def _common_cases(tables: dict[str, ResultsTable], mode: str) -> set[str]:
    sets = [t.case_set() for t in tables.values() if t.case_set()]
    if not sets:
        return set()
    return (set.intersection(*sets) if mode == "intersection"
            else set.union(*sets))


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.html:
        if args.runs or args.images or args.from_images:
            raise SystemExit("--html reports all canonical outputs; run/image selection flags apply only to plots")
        from src.reporting import write_experiment_report
        print(write_experiment_report(REPO_ROOT, repo_path(args.html)))
        return 0
    from src.plotter import BadPerformersTable, CVData, DiceDistributionPlot
    splits_path = repo_path(args.splits)
    gt_root = repo_path(args.gt_root)
    store = RunStore(splits_path=splits_path)
    runs = store.select(args.runs) if args.runs else store.discover()
    out_dir = Path(repo_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, ResultsTable] = {}
    used_metrics = False
    if not args.from_images:
        for run in runs:
            table = run.load_metrics()
            if table is not None:
                tables[run.run_id] = table
            else:
                print(f"warning: {run.run_id} has no metrics; skipped",
                      file=sys.stderr)
    run_names = {run.run_id: run.name for run in runs}

    if tables:
        used_metrics = True
        common = _common_cases(tables, args.case_set)
        if args.case_set == "intersection":
            dropped = {name: len(t.case_set() - common) for name, t in tables.items()}
            dropped = {k: v for k, v in dropped.items() if v}
            if dropped:
                print(f"using {len(common)} shared cases; excluded: " + ", ".join(
                    f"{k}:{v}" for k, v in dropped.items()))
            tables = {name: t.where(lambda r: r["case_id"] in common)
                      for name, t in tables.items()}
        display = {run_names.get(name, name): table for name, table in tables.items()}
        plot = DiceDistributionPlot(_EmptyCV(list(display)))
        plot.load_results(display)
    else:
        selected = [r for r in runs if r.pred_dir is not None]
        if not selected:
            raise SystemExit("no saved metrics and no prediction directories found")
        pairs = [(run_names.get(r.run_id, r.run_id), str(r.pred_dir)) for r in selected]
        cv = CVData(gt_root, splits_path, pairs)
        plot = DiceDistributionPlot(cv)
        display = {name: None for name, _ in pairs}

    plot.save(str(out_dir / "dice_distributions.png"))
    plot.save_by_run(str(out_dir), prefix="dice")

    if not args.no_bad_table:
        if args.images or args.from_images or not used_metrics:
            selected = [r for r in runs if r.pred_dir is not None]
            pairs = [(run_names.get(r.run_id, r.run_id), str(r.pred_dir))
                     for r in selected]
            if not pairs:
                raise SystemExit("--images requires prediction directories")
            cv = CVData(gt_root, splits_path, pairs)
            table = BadPerformersTable(cv, use_images=True)
        else:
            table = BadPerformersTable(None, use_images=False)
            table.load_results(tables)
        table.save(str(out_dir / "bad_performers.md"))
    return 0


class _EmptyCV:
    """Minimal stand-in when plots are filled from saved metrics."""

    def __init__(self, names):
        self.run_names = names


if __name__ == "__main__":
    raise SystemExit(main())
