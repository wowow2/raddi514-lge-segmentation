"""Model-agnostic evaluation and reporting for segmentation tasks."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import EVAL_LABELS, LABEL_NAMES
from src.datautils.emidec_loader import EMIDECCase
from src.datautils.metrics import SegmentationMetrics


@dataclass
class EvaluationReport:
    """Holds per-case and aggregated segmentation metrics."""

    per_case: list[dict[str, Any]] = field(default_factory=list)
    per_class: dict[str, dict[str, Any]] = field(default_factory=dict)
    overall: dict[str, Any] = field(default_factory=dict)

    def save_json(self, path: str | Path) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "per_case": self.per_case,
                    "per_class": self.per_class,
                    "overall": self.overall,
                },
                f,
                indent=2,
            )

    def to_markdown(self) -> str:
        header = "| Class | Dice mean | Dice std | HD95 mean | HD95 std |"
        divider = "|---|---|---|---|---|"
        lines = [header, divider]

        for str_lbl, pc in self.per_class.items():
            lbl = int(str_lbl)
            lines.append(
                f"| {LABEL_NAMES.get(lbl, str_lbl)} | {pc['dice_mean']:.3f} | {pc['dice_std']:.3f} "
                f"| {pc['hd95_mean']:.2f} | {pc['hd95_std']:.2f} |"
            )
        o = self.overall
        lines.append(
            f"| **All** | **{o['dice_mean']:.3f}** | **{o['dice_std']:.3f}** | **{o['hd95_mean']:.2f}** | — |"
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
        self.results: list[dict[str, Any]] = []

    def add_case(
        self,
        case_id: str,
        pred: np.ndarray,
        gt: np.ndarray,
        spacing: tuple[float, ...],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record evaluation for a single patient case."""
        pred = np.asarray(pred, dtype=np.uint8)
        gt = np.asarray(gt, dtype=np.uint8)

        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {case_id}: pred {pred.shape} vs gt {gt.shape}")

        eval_res = self.metrics.evaluate(pred, gt, spacing=spacing)

        entry = {
            "case_id": case_id,
            "dice": eval_res["dice"],
            "hd95": eval_res["hd95"],
            "gt_counts": {lbl: int((gt == lbl).sum()) for lbl in self.labels},
            **(metadata or {}),
        }
        self.results.append(entry)
        return entry

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
        )

    def aggregate(self) -> EvaluationReport:
        """Compute mean/std metrics."""
        summary = EvaluationReport(per_case=self.results)

        for lbl in self.labels:
            dsc = [c["dice"][lbl] for c in self.results]
            hd = [c["hd95"][lbl] for c in self.results if not np.isnan(c["hd95"][lbl])]
            n_gt = sum(1 for c in self.results if c["gt_counts"].get(lbl, 0) > 0)

            summary.per_class[str(lbl)] = {
                "dice_mean": float(np.mean(dsc)) if dsc else float("nan"),
                "dice_std": float(np.std(dsc)) if dsc else float("nan"),
                "hd95_mean": float(np.mean(hd)) if hd else float("nan"),
                "hd95_std": float(np.std(hd)) if hd else float("nan"),
                "n_dice": len(dsc),
                "n_hd95": len(hd),
                "n_gt_present": n_gt,
            }

        all_dsc = [c["dice"][lbl] for c in self.results for lbl in self.labels]
        all_hd = [c["hd95"][lbl] for c in self.results for lbl in self.labels if not np.isnan(c["hd95"][lbl])]

        summary.overall = {
            "dice_mean": float(np.mean(all_dsc)) if all_dsc else float("nan"),
            "dice_std": float(np.std(all_dsc)) if all_dsc else float("nan"),
            "hd95_mean": float(np.mean(all_hd)) if all_hd else float("nan"),
            "n_cases": len(self.results),
        }

        return summary