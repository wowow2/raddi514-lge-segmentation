"""Rebuild one ``run.json`` per historical experiment under ``var/runs``.

The manifests contain only facts derivable from directory names, paths, the
splits file, and the documented pipeline. Unknown fields stay empty and every
manifest is marked ``provenance.legacy`` + ``provenance.reconstructed``.

Examples::

    python scripts/rebuild_run_manifests.py              # dry run
    python scripts/rebuild_run_manifests.py --write      # write manifests
    python scripts/rebuild_run_manifests.py --write --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT
from src.rebuild import build_plan, rebuild_index, write_manifests
from src.runs import RunStore


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Rebuild per-experiment run.json manifests")
    parser.add_argument("--write", action="store_true",
                        help="write manifests (default: dry run)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing reconstructed manifests")
    parser.add_argument("--only", nargs="*", default=None,
                        help="limit to these run ids")
    parser.add_argument("--index", default="var/results/runs_index.json",
                        help="discovered-run snapshot to rewrite")
    parser.add_argument("--no-index", action="store_true",
                        help="do not rewrite the discovery snapshot")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    store = RunStore(repo_root=REPO_ROOT)
    plan = build_plan(REPO_ROOT)
    if args.only:
        wanted = set(args.only)
        plan = [run for run in plan if run.run_id in wanted]
        missing = sorted(wanted - {run.run_id for run in plan})
        if missing:
            print(f"unknown run id(s): {', '.join(missing)}", file=sys.stderr)
            return 1

    print(f"{'run_id':38s} {'stage':7s} {'metrics':34s} parent")
    for run in plan:
        metrics = str(run.metrics.relative_to(REPO_ROOT)) if run.metrics else "-"
        print(f"{run.run_id:38s} {run.stage or '-':7s} {metrics:34s} "
              f"{run.parent_run or '-'}")

    if not args.write:
        print(f"\n{len(plan)} run(s); dry run (pass --write to create manifests)")
        return 0

    result = write_manifests(plan, store, REPO_ROOT, force=args.force)
    print(f"\nwritten: {len(result['written'])} | skipped: "
          f"{len(result['skipped'])}")
    if result["skipped"]:
        print("skipped (existing non-legacy manifest): "
              + ", ".join(result["skipped"]))
    if not args.no_index:
        out = Path(args.index)
        count = rebuild_index(store, out)
        print(f"rebuilt {out} ({count} discovered run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
