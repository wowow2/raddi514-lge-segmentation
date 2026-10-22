"""Tidy results, schema-v1 compatibility, and run comparison."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.results import ResultsTable, compare, is_evaluation_report


def _row(run_id, case_id, cls, dice, hd95=None, fold=0, group="Pathological",
         gt=10, pred=10, tp=10, fp=0, fn=0):
    return {
        "run_id": run_id, "case_id": case_id, "fold": fold, "group": group,
        "class_id": cls, "class_name": {1: "Cavity", 2: "Myo", 3: "Scar", 4: "MVO"}[cls],
        "dice": dice, "dice_defined": True,
        "hd95": hd95, "hd95_defined": hd95 is not None,
        "hd95_reason": None if hd95 is not None else "gt_empty",
        "gt_voxels": gt, "pred_voxels": pred, "tp": tp, "fp": fp, "fn": fn,
        "gt_present": gt > 0, "pred_present": pred > 0,
    }


class ResultsTableTests(unittest.TestCase):
    def test_filter_and_summary(self):
        table = ResultsTable([
            _row("r", "Case_P001", 3, 0.5, 10.0),
            _row("r", "Case_P002", 3, 0.7, 12.0, fold=1),
            _row("r", "Case_N001", 3, 1.0, None, group="Normal", gt=0, pred=0),
        ])
        self.assertEqual(len(table.filter(class_id=3)), 3)
        self.assertEqual(len(table.filter(group="Normal")), 1)
        summary = table.summary("dice", by="class_id")
        self.assertEqual(summary[0]["n"], 3)
        self.assertAlmostEqual(summary[0]["mean"], (0.5 + 0.7 + 1.0) / 3)
        hd = table.summary("hd95", by="class_id")
        self.assertEqual(hd[0]["n"], 2)  # undefined value excluded

    def test_compare_intersection_and_missing(self):
        a = ResultsTable([_row("a", f"Case_P00{i}", 3, 0.4 + i / 10) for i in (1, 2, 3)])
        b = ResultsTable([_row("b", f"Case_P00{i}", 3, 0.6 + i / 10) for i in (2, 3, 4)])
        result = compare({"a": a, "b": b})
        self.assertEqual(result.case_ids, ["Case_P002", "Case_P003"])
        self.assertEqual(result.missing_by_run["a"], [])
        self.assertEqual(result.excluded_by_run["a"], ["Case_P001"])
        self.assertEqual(result.excluded_by_run["b"], ["Case_P004"])
        rows = result.to_rows()
        self.assertEqual(rows[0]["cases"], 2)
        self.assertEqual(rows[0]["excluded"], 1)

    def test_empty_run_makes_intersection_empty(self):
        a = ResultsTable([_row("a", "Case_P001", 3, 0.5)])
        empty = ResultsTable([])
        result = compare({"a": a, "empty": empty})
        self.assertEqual(result.case_ids, [])
        self.assertEqual(result.missing_by_run["a"], [])
        self.assertEqual(result.excluded_by_run["a"], ["Case_P001"])
        self.assertEqual(result.overall["a"]["n"], 0)

    def test_union_reports_missing_per_run(self):
        a = ResultsTable([_row("a", "Case_P001", 3, 0.5)])
        b = ResultsTable([_row("b", "Case_P002", 3, 0.5)])
        result = compare({"a": a, "b": b}, case_set="union")
        self.assertEqual(result.case_ids, ["Case_P001", "Case_P002"])
        self.assertEqual(result.missing_by_run["a"], ["Case_P002"])
        self.assertEqual(result.missing_by_run["b"], ["Case_P001"])
        self.assertEqual(result.excluded_by_run["a"], [])

    def test_require_same_cases_rejects_partial(self):
        a = ResultsTable([_row("a", "Case_P001", 3, 0.5)])
        b = ResultsTable([_row("b", "Case_P002", 3, 0.5)])
        with self.assertRaises(ValueError):
            compare({"a": a, "b": b}, require_same_cases=True)

    def test_fold_mismatch_rejected(self):
        a = ResultsTable([_row("a", "Case_P001", 3, 0.5, fold=0)])
        b = ResultsTable([_row("b", "Case_P001", 3, 0.5, fold=1)])
        with self.assertRaises(ValueError):
            compare({"a": a, "b": b})

    def test_join_case_metadata_by_id(self):
        table = ResultsTable([{**_row("r", "Case_P001", 3, 0.5), "group": None}])
        joined = table.join_cases({"Case_P001": {"group": "Pathological",
                                                 "spacing": [1.5, 1.5, 10.0]}})
        self.assertEqual(joined.rows[0]["group"], "Pathological")
        self.assertEqual(joined.rows[0]["spacing"], [1.5, 1.5, 10.0])

    def test_csv_roundtrip_header(self):
        table = ResultsTable([_row("r", "Case_P001", 3, 0.5, 10.0)])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.csv"
            table.to_csv(path)
            header = path.read_text().splitlines()[0].split(",")
            self.assertIn("tp", header)
            self.assertIn("hd95_reason", header)


class BackwardCompatibilityTests(unittest.TestCase):
    def _v1(self):
        return {
            "per_case": [
                {"case_id": "Case_P001", "fold": 0,
                 "dice": {"1": 0.9, "2": 0.8, "3": 0.5, "4": 1.0},
                 "hd95": {"1": 2.0, "2": 3.0, "3": 4.0, "4": math.nan},
                 "gt_counts": {"1": 10, "2": 20, "3": 0, "4": 0}},
                {"case_id": "Case_N002", "fold": 1,
                 "dice": {"1": 0.7, "2": 0.6, "3": 1.0, "4": 1.0},
                 "hd95": {"1": 5.0, "2": 6.0, "3": math.nan, "4": math.nan},
                 "gt_counts": {"1": 5, "2": 5, "3": 0, "4": 0}},
            ],
            "per_class": {"1": {"dice_mean": 0.8, "dice_std": 0.1,
                                "hd95_mean": 3.5, "hd95_std": 1.5}},
            "overall": {"dice_mean": 0.0, "dice_std": 0.0,
                        "hd95_mean": 0.0, "n_cases": 2},
        }

    def test_v1_rows_and_unknown_fields(self):
        table = ResultsTable.from_report(self._v1(), run_id="old")
        self.assertEqual(len(table), 8)
        scar = table.filter(case_id="Case_P001", class_id=3).rows[0]
        self.assertEqual(scar["dice"], 0.5)
        self.assertEqual(scar["gt_voxels"], 0)
        self.assertIsNone(scar["pred_voxels"])
        self.assertIsNone(scar["tp"])
        mvo = table.filter(case_id="Case_P001", class_id=4).rows[0]
        self.assertIsNone(mvo["hd95"])
        self.assertFalse(mvo["hd95_defined"])
        self.assertEqual(mvo["hd95_reason"], "unknown")

    def test_v1_group_derived_from_case_id(self):
        table = ResultsTable.from_report(self._v1(), run_id="old")
        self.assertEqual(table.filter(case_id="Case_P001").rows[0]["group"],
                         "Pathological")
        self.assertEqual(table.filter(case_id="Case_N002").rows[0]["group"], "Normal")

    def test_is_evaluation_report_rejects_diagnostics(self):
        self.assertFalse(is_evaluation_report({"per_case": {"Case_P001": {}}}))
        self.assertTrue(is_evaluation_report({"per_case": []}))
        self.assertTrue(is_evaluation_report({"results": []}))

    def test_saved_summary_matches_direct_metrics(self):
        from src.datautils.evaluator import EvaluationReport

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "v1.json"
            path.write_text(json.dumps(self._v1()))
            report = EvaluationReport.load(path)
            self.assertEqual(report.per_class["1"]["dice_mean"], 0.8)
            table = report.to_results_table()
            cavity = [r for r in table.summary("dice", by="class_id")
                      if r["key"] == 1][0]
            self.assertAlmostEqual(cavity["mean"], (0.9 + 0.7) / 2)
            self.assertEqual(report.schema_version, 1)


if __name__ == "__main__":
    unittest.main()
