"""Model-agnostic evaluation and reporting for segmentation tasks.

Schema v2 writes a tidy per-case/per-class ``results`` table, a case-level
``cases`` table (including the 5x5 confusion matrix), cached per-class
summaries, and the run metadata that produced it. Schema v1 files
(``per_case`` with nested ``dice``/``hd95`` dicts) remain readable.
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import EVAL_LABELS, LABEL_NAMES
from src.datautils.emidec_loader import EMIDECCase
from src.datautils.metrics import SegmentationMetrics
from src.results import ResultsTable

SCHEMA_VERSION = 2


def _json_safe(value: Any) -> Any:
    """Recursively replace NaN with None."""
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class EvaluationReport:
    """Per-case and aggregated segmentation metrics for one run."""

    run_id: str | None = None
    manifest: dict[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION
    cases: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    per_class: dict[str, dict[str, Any]] = field(default_factory=dict)
    overall: dict[str, Any] = field(default_factory=dict)
    empty_policy: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- legacy API
    @property
    def per_case(self) -> list[dict[str, Any]]:
        """Case-level view (schema-v1 style) reconstructed from result rows."""
        grouped: dict[str, dict[str, Any]] = {}
        for row in self.results:
            cid = row["case_id"]
            entry = grouped.setdefault(cid, {
                "case_id": cid, "fold": row.get("fold"),
                "group": row.get("group"), "dice": {}, "hd95": {},
                "gt_counts": {},
            })
            entry["dice"][row["class_id"]] = row.get("dice")
            entry["hd95"][row["class_id"]] = row.get("hd95")
            entry["gt_counts"][row["class_id"]] = row.get("gt_voxels")
        return list(grouped.values())

    # ------------------------------------------------------------------- I/O
    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "manifest": self.manifest,
            "empty_policy": self.empty_policy,
            "cases": self.cases,
            "results": self.results,
            "per_class": self.per_class,
            "overall": self.overall,
        })

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        os.makedirs(path.parent or Path("."), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path: str | Path, run_id: str | None = None) -> "EvaluationReport":
        with open(path) as f:
            data = json.load(f)
        if data.get("results"):
            return cls(
                run_id=data.get("run_id") or run_id or Path(path).stem,
                manifest=data.get("manifest"),
                schema_version=data.get("schema_version", SCHEMA_VERSION),
                cases=data.get("cases", []),
                results=data.get("results", []),
                per_class=data.get("per_class", {}),
                overall=data.get("overall", {}),
                empty_policy=data.get("empty_policy", {}),
            )

        # Schema v1: convert rows but keep the stored aggregates untouched.
        table = ResultsTable.from_report(data, run_id=run_id or Path(path).stem)
        cases: dict[str, dict] = {}
        for row in table.rows:
            entry = cases.setdefault(row["case_id"], {
                "case_id": row["case_id"], "fold": row.get("fold"),
                "group": row.get("group"), "gt_counts": {},
            })
            if row.get("gt_voxels") is not None:
                entry["gt_counts"][str(row["class_id"])] = row["gt_voxels"]
        return cls(
            run_id=run_id or Path(path).stem,
            schema_version=1,
            cases=list(cases.values()),
            results=table.rows,
            per_class=data.get("per_class", {}),
            overall=data.get("overall", {}),
            empty_policy={"dice": "unknown", "hd95": "unknown",
                          "source": "schema-v1 file"},
        )

    def to_results_table(self) -> ResultsTable:
        return ResultsTable(self.results)

    # -------------------------------------------------------------- rendering
    @staticmethod
    def _fmt(value: Any, digits: int) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        return f"{value:.{digits}f}"

    def to_markdown(self) -> str:
        header = "| Class | Dice mean | Dice std | HD95 mean | HD95 std |"
        divider = "|---|---|---|---|---|"
        lines = [header, divider]

        for str_lbl, pc in self.per_class.items():
            lbl = int(str_lbl)
            lines.append(
                f"| {LABEL_NAMES.get(lbl, str_lbl)} | {self._fmt(pc['dice_mean'], 3)} "
                f"| {self._fmt(pc['dice_std'], 3)} | {self._fmt(pc['hd95_mean'], 2)} "
                f"| {self._fmt(pc['hd95_std'], 2)} |"
            )
        o = self.overall
        lines.append(
            f"| **All** | **{self._fmt(o['dice_mean'], 3)}** "
            f"| **{self._fmt(o['dice_std'], 3)}** "
            f"| **{self._fmt(o['hd95_mean'], 2)}** | — |"
        )
        return "\n".join(lines)


class DatasetEvaluator:
    """Evaluates arbitrary segmentation predictions and probabilities against ground truth."""

    def __init__(
        self,
        labels: tuple[int, ...] = EVAL_LABELS,
        metrics: SegmentationMetrics | None = None,
    ):
        self.labels = tuple(labels)
        self.metrics = metrics or SegmentationMetrics(self.labels)
        self.cases: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []

    def add_case(
        self,
        case_id: str,
        pred: np.ndarray,
        gt: np.ndarray,
        spacing: tuple[float, ...],
        metadata: dict[str, Any] | None = None,
        group: str | None = None,
    ) -> dict[str, Any]:
        """Record evaluation for a single patient case."""
        pred = np.asarray(pred, dtype=np.uint8)
        gt = np.asarray(gt, dtype=np.uint8)

        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {case_id}: pred {pred.shape} vs gt {gt.shape}")

        meta = dict(metadata or {})
        group = group if group is not None else meta.pop("group", None)
        fold = meta.pop("fold", None)
        for reserved in ("case_id", "group", "spacing", "gt_counts",
                         "pred_counts", "confusion"):
            meta.pop(reserved, None)
        eval_res = self.metrics.evaluate(pred, gt, spacing=spacing)
        confusion = self.metrics.confusion(pred, gt, n_labels=5)
        gt_counts = {lbl: int(confusion[lbl, :].sum()) for lbl in self.labels}
        pred_counts = {lbl: int(confusion[:, lbl].sum()) for lbl in self.labels}

        case_entry: dict[str, Any] = {
            "case_id": case_id,
            "fold": fold,
            "group": group,
            "spacing": [float(s) for s in spacing],
            "gt_counts": gt_counts,
            "pred_counts": pred_counts,
            "confusion": confusion.ravel().tolist(),  # 5x5, rows=gt, cols=pred
            **meta,
        }
        self.cases.append(case_entry)

        for lbl in self.labels:
            tp = int(confusion[lbl, lbl])
            fp = pred_counts[lbl] - tp
            fn = gt_counts[lbl] - tp
            hd = eval_res["hd95"][lbl]
            hd_defined = not (isinstance(hd, float) and math.isnan(hd))
            gt_present = gt_counts[lbl] > 0
            pred_present = pred_counts[lbl] > 0
            if hd_defined:
                reason = None
            elif not gt_present and not pred_present:
                reason = "both_empty"
            elif not gt_present:
                reason = "gt_empty"
            elif not pred_present:
                reason = "pred_empty"
            else:
                reason = "undefined"

            self.results.append({
                "run_id": None,
                "case_id": case_id,
                "fold": fold,
                "group": group,
                "class_id": lbl,
                "class_name": LABEL_NAMES.get(lbl, str(lbl)),
                "dice": float(eval_res["dice"][lbl]),
                "dice_defined": True,
                "hd95": None if not hd_defined else float(hd),
                "hd95_defined": hd_defined,
                "hd95_reason": reason,
                "gt_voxels": gt_counts[lbl],
                "pred_voxels": pred_counts[lbl],
                "tp": tp, "fp": fp, "fn": fn,
                "gt_present": gt_present,
                "pred_present": pred_present,
            })
        return self.results[-len(self.labels)]

    def add_emidec_case(
        self,
        case: EMIDECCase,
        pred: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _, gt = case.load()
        if gt is None:
            raise ValueError(f"Case {case.case_id} has no ground truth contours.")
        if case.zooms is None:
            raise ValueError(f"Case {case.case_id} has uninitialized voxel zooms.")

        return self.add_case(
            case_id=case.case_id,
            pred=pred,
            gt=gt,
            spacing=tuple(case.zooms[:3]),
            metadata=metadata,
            group=case.group,
        )

    def aggregate(self, run_id: str | None = None,
                  manifest: dict[str, Any] | None = None) -> EvaluationReport:
        """Compute mean/std metrics and the tidy table."""
        summary = EvaluationReport(
            run_id=run_id,
            manifest=manifest,
            cases=self.cases,
            results=[dict(row, run_id=run_id) for row in self.results],
            empty_policy={
                "dice": "both masks empty -> 1.0; one mask empty -> 0.0",
                "hd95": "undefined (null) when either mask is empty",
                "labels": list(self.labels),
            },
        )

        for lbl in self.labels:
            dsc = [c["dice"][lbl] for c in summary.per_case]
            hd = [c["hd95"][lbl] for c in summary.per_case
                  if c["hd95"][lbl] is not None
                  and not (isinstance(c["hd95"][lbl], float)
                           and math.isnan(c["hd95"][lbl]))]
            n_gt = sum(1 for c in summary.per_case if c["gt_counts"].get(lbl, 0) > 0)

            summary.per_class[str(lbl)] = {
                "dice_mean": float(np.mean(dsc)) if dsc else float("nan"),
                "dice_std": float(np.std(dsc)) if dsc else float("nan"),
                "hd95_mean": float(np.mean(hd)) if hd else float("nan"),
                "hd95_std": float(np.std(hd)) if hd else float("nan"),
                "n_dice": len(dsc),
                "n_hd95": len(hd),
                "n_gt_present": n_gt,
            }

        all_dsc = [c["dice"][lbl] for c in summary.per_case for lbl in self.labels]
        all_hd = [c["hd95"][lbl] for c in summary.per_case for lbl in self.labels
                  if c["hd95"][lbl] is not None
                  and not (isinstance(c["hd95"][lbl], float)
                           and math.isnan(c["hd95"][lbl]))]

        summary.overall = {
            "dice_mean": float(np.mean(all_dsc)) if all_dsc else float("nan"),
            "dice_std": float(np.std(all_dsc)) if all_dsc else float("nan"),
            "hd95_mean": float(np.mean(all_hd)) if all_hd else float("nan"),
            "n_cases": len(summary.per_case),
        }

        return summary
