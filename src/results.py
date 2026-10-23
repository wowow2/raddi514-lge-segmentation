"""Tidy per-case, per-class results plus run comparison.

Canonical metric file (schema v2) stores one row per ``run x case x class``::

    {"run_id": ..., "case_id": ..., "fold": 0, "group": "Pathological",
     "class_id": 3, "class_name": "Scar", "dice": 0.61, "dice_defined": true,
     "hd95": 12.3, "hd95_defined": true, "hd95_reason": null,
     "gt_voxels": 900, "pred_voxels": 700, "tp": 650, "fp": 50, "fn": 250,
     "gt_present": true, "pred_present": true}

Schema-v1 files (``per_case`` with nested ``dice``/``hd95`` dicts) remain
readable. Fields that cannot be recovered are ``None`` and marked unknown
rather than guessed.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.config import EVAL_LABELS, LABEL_NAMES

RESULT_COLUMNS = (
    "run_id", "case_id", "fold", "group", "class_id", "class_name",
    "dice", "dice_defined", "hd95", "hd95_defined", "hd95_reason",
    "gt_voxels", "pred_voxels", "tp", "fp", "fn", "gt_present", "pred_present",
)

METRIC_KEYS = {"dice": "dice_defined", "hd95": "hd95_defined"}


def is_evaluation_report(data: Mapping[str, Any]) -> bool:
    """True for metric files (v1 or v2), false for unrelated JSON such as diagnostics."""
    if not isinstance(data, Mapping):
        return False
    return isinstance(data.get("results"), list) or isinstance(data.get("per_case"), list)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and math.isnan(value))


def _clean(value: Any) -> Any:
    """Turn NaN into None so JSON stays standards-compliant."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class ResultsTable:
    """A list of tidy result rows with simple filtering and summaries."""

    def __init__(self, rows: Iterable[dict] = ()):
        self.rows: list[dict] = [dict(r) for r in rows]

    # ------------------------------------------------------------------ load
    @classmethod
    def from_report(cls, report: Mapping[str, Any],
                    run_id: str | None = None) -> "ResultsTable":
        """Read a v2 report, or convert a legacy v1 ``per_case`` report."""
        run_id = run_id or report.get("run_id")
        if report.get("results"):
            rows = []
            for raw in report["results"]:
                row = {col: _clean(raw.get(col)) for col in RESULT_COLUMNS}
                row["run_id"] = raw.get("run_id") or run_id
                if row.get("dice_defined") is None:
                    row["dice_defined"] = row.get("dice") is not None
                if row.get("hd95_defined") is None:
                    row["hd95_defined"] = row.get("hd95") is not None
                rows.append(row)
            return cls(rows)
        return cls(_rows_from_v1(report, run_id))

    @classmethod
    def load(cls, path: str | Path, run_id: str | None = None) -> "ResultsTable":
        with open(path) as f:
            report = json.load(f)
        return cls.from_report(report, run_id=run_id or Path(path).stem)

    # --------------------------------------------------------------- filters
    def filter(self, **equals: Any) -> "ResultsTable":
        rows = self.rows
        for key, value in equals.items():
            rows = [r for r in rows if r.get(key) == value]
        return ResultsTable(rows)

    def where(self, predicate: Callable[[dict], bool]) -> "ResultsTable":
        return ResultsTable(r for r in self.rows if predicate(r))

    def classes(self) -> list[int]:
        return sorted({r["class_id"] for r in self.rows if r.get("class_id") is not None})

    def case_ids(self) -> list[str]:
        return sorted({r["case_id"] for r in self.rows})

    def case_set(self) -> set[str]:
        return {r["case_id"] for r in self.rows}

    def fold_by_case(self) -> dict[str, int]:
        return {r["case_id"]: r.get("fold") for r in self.rows}

    def join_cases(self, cases: Mapping[str, dict]) -> "ResultsTable":
        """Fill missing group/geometry fields from a case metadata table."""
        rows = []
        for row in self.rows:
            meta = cases.get(row["case_id"], {})
            merged = dict(row)
            if merged.get("group") is None:
                merged["group"] = meta.get("group")
            for key in ("shape", "spacing", "n_slices", "has_gt"):
                if merged.get(key) is None and key in meta:
                    merged[key] = meta[key]
            rows.append(merged)
        return ResultsTable(rows)

    def is_empty(self) -> bool:
        return not self.rows

    def __len__(self) -> int:
        return len(self.rows)

    # ------------------------------------------------------------- summaries
    def summary(self, metric: str = "dice", cases: Iterable[str] | None = None,
                by: str = "class_name") -> list[dict]:
        """Mean/std/n of a metric per class (or another key)."""
        defined_key = METRIC_KEYS[metric]
        case_set = set(cases) if cases is not None else None
        groups: dict[Any, list[float]] = {}
        for row in self.rows:
            if case_set is not None and row["case_id"] not in case_set:
                continue
            if not row.get(defined_key):
                continue
            value = row.get(metric)
            if not _is_number(value):
                continue
            groups.setdefault(row.get(by), []).append(float(value))
        return [
            {"key": key, "n": len(vals),
             "mean": sum(vals) / len(vals),
             "std": _std(vals)}
            for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0]))
        ]

    def overall(self, metric: str = "dice",
                cases: Iterable[str] | None = None) -> dict:
        defined_key = METRIC_KEYS[metric]
        case_set = set(cases) if cases is not None else None
        values = [
            float(r[metric]) for r in self.rows
            if (case_set is None or r["case_id"] in case_set)
            and r.get(defined_key) and _is_number(r.get(metric))
        ]
        return {"n": len(values),
                "mean": sum(values) / len(values) if values else None,
                "std": _std(values)}

    # ------------------------------------------------------------------ misc
    def to_csv(self, path: str | Path, columns: Iterable[str] = RESULT_COLUMNS) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = list(columns)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: row.get(k) for k in columns})
        return str(path)

    def __repr__(self) -> str:
        return f"ResultsTable(rows={len(self.rows)}, cases={len(self.case_ids())})"


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def group_from_case_id(case_id: str | None) -> str | None:
    """Dataset convention (same rule as the loader): P -> Pathological, N -> Normal."""
    if not case_id:
        return None
    if "Case_P" in case_id:
        return "Pathological"
    if "Case_N" in case_id:
        return "Normal"
    return None


def _rows_from_v1(report: Mapping[str, Any], run_id: str | None) -> list[dict]:
    """Best-effort conversion of schema-v1 ``per_case`` metrics."""
    rows: list[dict] = []
    for entry in report.get("per_case", []):
        gt_counts = entry.get("gt_counts") or {}
        for raw_lbl, dice in (entry.get("dice") or {}).items():
            lbl = int(raw_lbl)
            hd_raw = (entry.get("hd95") or {}).get(raw_lbl)
            hd = None if hd_raw is None or (
                isinstance(hd_raw, float) and math.isnan(hd_raw)) else float(hd_raw)
            gt_vox = gt_counts.get(raw_lbl)
            rows.append({
                "run_id": run_id,
                "case_id": entry.get("case_id"),
                "fold": entry.get("fold"),
                "group": entry.get("group") or group_from_case_id(entry.get("case_id")),
                "class_id": lbl,
                "class_name": LABEL_NAMES.get(lbl, str(lbl)),
                "dice": float(dice) if dice is not None else None,
                "dice_defined": dice is not None,
                "hd95": hd,
                "hd95_defined": hd is not None,
                "hd95_reason": None if hd is not None else "unknown",
                "gt_voxels": gt_vox,
                "pred_voxels": None,
                "tp": None, "fp": None, "fn": None,
                "gt_present": None if gt_vox is None else gt_vox > 0,
                "pred_present": None,
            })
    return rows


# ---------------------------------------------------------------- comparison
@dataclass
class Comparison:
    metric: str
    case_set: str
    runs: list[str]
    case_ids: list[str]
    cases_by_run: dict[str, set[str]]
    missing_by_run: dict[str, list[str]]
    excluded_by_run: dict[str, list[str]] = field(default_factory=dict)
    summaries: dict[str, dict[str, dict]] = field(default_factory=dict)
    overall: dict[str, dict] = field(default_factory=dict)

    def to_rows(self) -> list[dict]:
        """One row per run: per-class means plus an overall column."""
        classes = [lbl for lbl in EVAL_LABELS]
        rows = []
        for run in self.runs:
            row = {"run_id": run, "cases": len(self.case_ids),
                   "missing": len(self.missing_by_run.get(run, [])),
                   "excluded": len(self.excluded_by_run.get(run, []))}
            per = self.summaries.get(run, {})
            for lbl in classes:
                value = per.get(lbl, {}).get("mean")
                row[LABEL_NAMES[lbl]] = value
            row["All"] = self.overall.get(run, {}).get("mean")
            rows.append(row)
        return rows

    def to_markdown(self) -> str:
        rows = self.to_rows()
        if not rows:
            return "(no results)"
        headers = list(rows[0])
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join(["---"] * len(headers)) + "|"]
        for row in rows:
            cells = []
            for key in headers:
                value = row[key]
                if isinstance(value, float):
                    cells.append(f"{value:.3f}")
                else:
                    cells.append("—" if value is None else str(value))
            lines.append("| " + " | ".join(cells) + " |")
        label = f"{self.metric} on {self.case_set} case set"
        return f"_{label} (n={len(self.case_ids)})_\n\n" + "\n".join(lines)

    def to_csv(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_rows()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        return str(path)


def compare(tables: Mapping[str, ResultsTable], metric: str = "dice",
            case_set: str = "intersection",
            require_same_cases: bool = False) -> Comparison:
    """Compare runs on the intersection (default) or union of their cases.

    ``selected`` is the intersection/union of *all* run case sets, including
    empty runs. ``missing_by_run`` is ``selected - run_cases`` (cases the
    comparison needs but the run lacks); ``excluded_by_run`` is
    ``run_cases - selected`` (cases the run has but that are outside the
    comparison set).

    Raises ``ValueError`` when ``require_same_cases`` is set and runs cover
    different case sets, or when a case is assigned to different folds.
    """
    if metric not in METRIC_KEYS:
        raise ValueError(f"metric must be one of {sorted(METRIC_KEYS)}")
    if case_set not in ("intersection", "union"):
        raise ValueError("case_set must be 'intersection' or 'union'")

    runs = list(tables)
    cases_by_run = {run: tables[run].case_set() for run in runs}
    sets = list(cases_by_run.values())
    if not sets:
        selected: set[str] = set()
    elif case_set == "intersection":
        selected = set.intersection(*sets)
    else:
        selected = set.union(*sets)

    if require_same_cases and runs:
        reference = cases_by_run[runs[0]]
        different = [run for run in runs if cases_by_run[run] != reference]
        if different:
            raise ValueError(
                "Runs cover different case sets; refusing to compare: "
                + ", ".join(different))

    # Fold consistency: the same case must belong to the same fold everywhere.
    for run in runs:
        for case_id, fold in tables[run].fold_by_case().items():
            for other in runs:
                if other == run:
                    continue
                other_fold = tables[other].fold_by_case().get(case_id)
                if other_fold is not None and fold is not None and other_fold != fold:
                    raise ValueError(
                        f"{case_id} is fold {fold} in {run} but fold "
                        f"{other_fold} in {other}; split identities differ")

    result = Comparison(
        metric=metric, case_set=case_set, runs=runs, case_ids=sorted(selected),
        cases_by_run=cases_by_run,
        missing_by_run={run: sorted(selected - cases_by_run[run]) for run in runs},
        excluded_by_run={run: sorted(cases_by_run[run] - selected) for run in runs},
    )
    for run in runs:
        per = {row["key"]: row for row in tables[run].summary(
            metric, cases=selected, by="class_id")}
        result.summaries[run] = per
        result.overall[run] = tables[run].overall(metric, cases=selected)
    return result
