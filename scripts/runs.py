"""Query runs and compare saved results.

Examples::

    python scripts/runs.py list
    python scripts/runs.py list --tag baseline --model polar_unet
    python scripts/runs.py show polar_add_r64_pp
    python scripts/runs.py validate
    python scripts/runs.py results --all --class-name Scar --gt-present
    python scripts/runs.py compare --all --metric dice --require-same-cases
    python scripts/runs.py compare 501_regions_pp polar_add_r64_pp --csv /tmp/cmp.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.results import ResultsTable, compare
from src.runs import RunStore


def _store(args) -> RunStore:
    return RunStore(splits_path=args.splits)


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no runs)")
        return
    headers = list(rows[0])
    widths = {h: max(len(str(h)), *(len(str(r[h])) for r in rows)) for h in headers}
    print("  ".join(str(h).ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row[h]).ljust(widths[h]) for h in headers))


def _select(args, store: RunStore):
    filters = {
        "tag": args.tag,
        "model": args.model,
        "stage": args.stage,
        "parent_run": args.parent,
        "complete": (True if args.complete else
                     False if getattr(args, "partial", False) else None),
        "has_metrics": True if args.has_metrics else None,
    }
    return store.select(getattr(args, "runs", None) or None, **filters)


def cmd_list(args) -> int:
    store = _store(args)
    runs = _select(args, store)
    if args.json:
        print(json.dumps([run.manifest.to_dict() for run in runs], indent=2))
        return 0
    _print_table([run.summary_row() for run in runs])
    return 0


def cmd_show(args) -> int:
    run = _store(args).get(args.run_id)
    print(json.dumps(run.manifest.to_dict(), indent=2))
    print("\nStatus:")
    print(json.dumps(run.status_summary(), indent=2, default=str))
    return 0


def cmd_validate(args) -> int:
    from src.layout import check_outputs

    try:
        experiments, outputs = check_outputs(Path(__file__).resolve().parent.parent)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"ok: {experiments} experiments, {outputs} outputs")
    return 0


def _row_matches(row: dict, args) -> bool:
    """Filters for the ``results`` subcommand."""
    if args.class_name:
        wanted = args.class_name.lower()
        if str(row.get("class_name", "")).lower() != wanted:
            return False
    if args.group and row.get("group") != args.group:
        return False
    if args.fold is not None and row.get("fold") != args.fold:
        return False
    if args.gt_present and not row.get("gt_present"):
        return False
    if args.pred_empty and row.get("pred_voxels") != 0:
        return False
    if args.fp_min is not None and (row.get("fp") or 0) < args.fp_min:
        return False
    if args.fn_min is not None and (row.get("fn") or 0) < args.fn_min:
        return False
    return True


def _load_tables(store: RunStore, run_ids, args) -> dict[str, ResultsTable]:
    if run_ids:
        runs = store.select(run_ids)
    else:
        runs = [r for r in _select(args, store) if r.metrics_path]
    tables: dict[str, ResultsTable] = {}
    for run in runs:
        table = run.load_metrics()
        if table is None:
            print(f"warning: {run.run_id} has no metrics file; skipped", file=sys.stderr)
            continue
        tables[run.run_id] = table
    if not tables:
        raise SystemExit("no saved metrics found for the selected runs")
    return tables


def cmd_results(args) -> int:
    store = _store(args)
    tables = _load_tables(store, args.runs, args)
    rows: list[dict] = []
    for run_id, table in tables.items():
        rows.extend(dict(row, run_id=run_id) for row in table.rows)

    rows = [row for row in rows if _row_matches(row, args)]
    if args.csv:
        ResultsTable(rows).to_csv(args.csv)
        print(f"wrote {len(rows)} rows -> {args.csv}")
        return 0

    for run_id in tables:
        table = ResultsTable([r for r in rows if r["run_id"] == run_id])
        if table.is_empty():
            continue
        print(f"\n[{run_id}] {len(table.rows)} rows, {len(table.case_ids())} cases")
        for metric in ("dice", "hd95"):
            summary = table.overall(metric)
            if summary["n"]:
                print(f"  {metric}: mean {summary['mean']:.3f} over {summary['n']} values")
    limit = len(rows) if args.print_all else args.limit
    for row in rows[:limit]:
        print(json.dumps({k: row[k] for k in
                          ("run_id", "case_id", "fold", "class_name", "dice",
                           "hd95", "gt_voxels", "pred_voxels", "tp", "fp", "fn")
                          if k in row}, default=str))
    if limit < len(rows):
        print(f"... {len(rows) - limit} more rows (use --print-all or --csv)")
    return 0


def cmd_compare(args) -> int:
    store = _store(args)
    run_ids = args.runs or None
    if not run_ids and not (args.tag or args.model or args.stage or args.all):
        raise SystemExit("pass run ids or --all/--tag/--model/--stage")
    if not run_ids:
        selected = [r for r in _select(args, store) if r.metrics_path]
        skipped = [r.run_id for r in _select(args, store) if not r.metrics_path]
        if skipped:
            print(f"warning: no metrics for {', '.join(skipped)}; skipped",
                  file=sys.stderr)
        run_ids = [r.run_id for r in selected]
    runs = store.select(run_ids)
    if not args.allow_mixed_splits:
        hashes = {run.run_id: run.manifest.get("data.splits_sha256") for run in runs}
        known = {value for value in hashes.values() if value}
        if len(known) > 1:
            detail = ", ".join(f"{k}={v[:8]}" for k, v in hashes.items() if v)
            raise SystemExit(f"Runs used different splits: {detail}")
    tables: dict[str, ResultsTable] = {}
    for run in runs:
        table = run.load_metrics()
        if table is None:
            raise SystemExit(f"{run.run_id} has no metrics file")
        tables[run.run_id] = table
    try:
        comparison = compare(
            tables, metric=args.metric, case_set=args.case_set,
            require_same_cases=args.require_same_cases)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))
    print(comparison.to_markdown())
    missing = {run: ids for run, ids in comparison.missing_by_run.items() if ids}
    excluded = {run: ids for run, ids in comparison.excluded_by_run.items() if ids}
    if missing:
        print("\nMissing from the comparison (case absent in that run):")
        for run, ids in missing.items():
            print(f"  {run}: {len(ids)} case(s) "
                  f"({', '.join(ids[:5])}{'...' if len(ids) > 5 else ''})")
    if excluded:
        print("\nExcluded from the comparison (case present but outside the set):")
        for run, ids in excluded.items():
            print(f"  {run}: {len(ids)} case(s) "
                  f"({', '.join(ids[:5])}{'...' if len(ids) > 5 else ''})")
    if args.csv:
        print(f"\nwrote -> {comparison.to_csv(args.csv)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover runs and compare saved metrics")
    parser.add_argument("--splits", default=None, help="default splits_final.json")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_selection(p):
        p.add_argument("--tag")
        p.add_argument("--model")
        p.add_argument("--stage", choices=["stage1", "stage2"])
        p.add_argument("--parent")
        p.add_argument("--complete", action="store_true")
        p.add_argument("--partial", action="store_true",
                       help="only runs with missing expected outputs")
        p.add_argument("--has-metrics", action="store_true")

    p = sub.add_parser("list", help="list discovered runs")
    add_selection(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="show one run's manifest and status")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("validate", help="check every output's split and OOF coverage")
    p.add_argument("--all", action="store_true", help="accepted for compatibility")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("results", help="print/filter tidy result rows")
    add_selection(p)
    p.add_argument("runs", nargs="*")
    p.add_argument("--all", action="store_true", help="use every selected run")
    p.add_argument("--class-name", dest="class_name")
    p.add_argument("--group", choices=["Pathological", "Normal", "Unknown"])
    p.add_argument("--fold", type=int)
    p.add_argument("--gt-present", action="store_true")
    p.add_argument("--pred-empty", action="store_true",
                   help="only rows where pred_voxels is exactly 0 "
                        "(unknown counts are excluded)")
    p.add_argument("--fp-min", type=int, default=None,
                   help="only rows with at least N false positives")
    p.add_argument("--fn-min", type=int, default=None,
                   help="only rows with at least N false negatives")
    p.add_argument("--csv")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--print-all", action="store_true")
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("compare", help="compare saved metrics across runs")
    add_selection(p)
    p.add_argument("runs", nargs="*")
    p.add_argument("--all", action="store_true", help="compare every selected run")
    p.add_argument("--metric", choices=["dice", "hd95"], default="dice")
    p.add_argument("--case-set", choices=["intersection", "union"],
                   default="intersection")
    p.add_argument("--require-same-cases", action="store_true",
                   help="reject runs that do not cover identical case sets")
    p.add_argument("--allow-mixed-splits", action="store_true")
    p.add_argument("--csv")
    p.set_defaults(func=cmd_compare)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
