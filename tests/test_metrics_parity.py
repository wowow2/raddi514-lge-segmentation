"""Confusion counts and metric parity for synthetic arrays."""

import math
import unittest

import numpy as np

from src.datautils.evaluator import DatasetEvaluator, EvaluationReport
from src.datautils.metrics import SegmentationMetrics


class ConfusionTests(unittest.TestCase):
    def test_manual_counts(self):
        gt = np.array([[0, 1], [3, 3]], dtype=np.uint8)
        pred = np.array([[0, 2], [3, 0]], dtype=np.uint8)
        conf = SegmentationMetrics().confusion(pred, gt)
        self.assertEqual(conf[0, 0], 1)   # gt bg -> pred bg
        self.assertEqual(conf[1, 2], 1)   # gt cavity -> pred myo
        self.assertEqual(conf[3, 3], 1)   # gt scar -> pred scar
        self.assertEqual(conf[3, 0], 1)   # gt scar -> pred bg
        self.assertEqual(int(conf.sum()), gt.size)

    def test_dice_from_confusion_matches_metric(self):
        rng = np.random.default_rng(7)
        gt = rng.integers(0, 5, size=(9, 9, 2)).astype(np.uint8)
        pred = rng.integers(0, 5, size=(9, 9, 2)).astype(np.uint8)
        metrics = SegmentationMetrics()
        conf = metrics.confusion(pred, gt)
        dice = metrics.dice(pred, gt)
        for lbl in (1, 2, 3, 4):
            tp = conf[lbl, lbl]
            fp = int(conf[:, lbl].sum()) - tp
            fn = int(conf[lbl, :].sum()) - tp
            expected = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
            self.assertAlmostEqual(dice[lbl], expected, places=12)

    def test_rejects_out_of_range_labels(self):
        with self.assertRaises(ValueError):
            SegmentationMetrics().confusion(
                np.array([5], dtype=np.uint8), np.array([0], dtype=np.uint8))


class EvaluatorTests(unittest.TestCase):
    def _evaluate(self, pred, gt):
        ev = DatasetEvaluator()
        ev.add_case("Case_P001", pred, gt, spacing=(1.0, 1.0, 10.0),
                    metadata={"fold": 2}, group="Pathological")
        return ev.aggregate(run_id="r")

    def test_aggregate_matches_direct_metrics(self):
        rng = np.random.default_rng(3)
        gt = rng.integers(0, 5, size=(7, 7, 3)).astype(np.uint8)
        pred = rng.integers(0, 5, size=(7, 7, 3)).astype(np.uint8)
        report = self._evaluate(pred, gt)
        direct = SegmentationMetrics().evaluate(pred, gt, (1.0, 1.0, 10.0))
        for lbl in (1, 2, 3, 4):
            self.assertAlmostEqual(report.per_class[str(lbl)]["dice_mean"],
                                   direct["dice"][lbl], places=12)
            if not math.isnan(direct["hd95"][lbl]):
                self.assertAlmostEqual(report.per_class[str(lbl)]["hd95_mean"],
                                       direct["hd95"][lbl], places=12)

    def test_empty_reasons_and_conventions(self):
        gt = np.zeros((4, 4, 1), dtype=np.uint8)
        pred = np.zeros_like(gt)
        pred[0, 0] = 3
        report = self._evaluate(pred, gt)
        rows = {(r["class_id"], r["case_id"]): r for r in report.results}
        scar = rows[(3, "Case_P001")]
        self.assertEqual(scar["dice"], 0.0)          # one mask empty
        self.assertFalse(scar["hd95_defined"])
        self.assertEqual(scar["hd95_reason"], "gt_empty")
        self.assertFalse(scar["gt_present"])
        self.assertTrue(scar["pred_present"])
        myo = rows[(2, "Case_P001")]
        self.assertEqual(myo["dice"], 1.0)           # both empty
        self.assertEqual(myo["hd95_reason"], "both_empty")

    def test_aggregate_counts_tp_fp_fn(self):
        gt = np.array([[[3], [3]], [[3], [0]]], dtype=np.uint8)
        pred = np.array([[[3], [2]], [[0], [0]]], dtype=np.uint8)
        report = self._evaluate(pred, gt)
        scar = [r for r in report.results if r["class_id"] == 3][0]
        self.assertEqual((scar["tp"], scar["fp"], scar["fn"]), (1, 0, 2))
        self.assertEqual(scar["pred_voxels"], 1)

    def test_save_roundtrip_has_no_nan(self):
        import tempfile
        from pathlib import Path

        gt = np.zeros((3, 3, 1), dtype=np.uint8)
        report = self._evaluate(gt, gt)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.json"
            report.save_json(path)
            text = path.read_text()
            self.assertNotIn("NaN", text)
            loaded = EvaluationReport.load(path)
            self.assertEqual(len(loaded.results), len(report.results))
            self.assertEqual(loaded.run_id, "r")


if __name__ == "__main__":
    unittest.main()
