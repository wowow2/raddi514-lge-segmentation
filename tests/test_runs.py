"""Run discovery, manifests, and lightweight status summaries."""

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from src.layout import write_json
from src.runs import RunManifest, RunStore


def write_splits(root: Path) -> Path:
    path = root / "splits.json"
    path.write_text(json.dumps([
        {"train": ["EMIDEC_P002"], "val": ["EMIDEC_P001"]},
        {"train": ["EMIDEC_P001"], "val": ["EMIDEC_P002"]},
    ]))
    return path


def write_prediction(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 1), dtype=np.uint8), np.eye(4)),
             str(path))


def write_manifest(directory: Path, run_id: str, **fields) -> Path:
    root = directory.parent
    path = root / "experiments" / run_id / "outputs" / run_id / "output.json"
    write_json(path, {"run_id": run_id, **fields})
    write_json(path.parents[2] / "experiment.json", {
        "experiment_id": run_id, "outputs": [run_id], "default_output": run_id})
    return path


def write_metrics(path: Path, run_id: str, cases) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id, fold in cases:
        for cls in (1, 2, 3, 4):
            rows.append({
                "run_id": run_id, "case_id": case_id, "fold": fold,
                "group": "Pathological", "class_id": cls, "class_name": "x",
                "dice": 0.5, "dice_defined": True,
                "hd95": None, "hd95_defined": False, "hd95_reason": "gt_empty",
                "gt_voxels": 0, "pred_voxels": 0, "tp": 0, "fp": 0, "fn": 0,
                "gt_present": False, "pred_present": False,
            })
    path.write_text(json.dumps({"schema_version": 2, "run_id": run_id,
                                "results": rows, "per_class": {}, "overall": {}}))


class RunStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.splits = write_splits(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _store(self):
        return RunStore(repo_root=self.root, splits_path=self.splits)

    def test_canonical_discovery_and_filters(self):
        write_manifest(self.root / "a", "run1", stage="stage1", tags=["baseline"], model={"name": "unet"})
        write_manifest(self.root / "b", "run2", stage="stage2", model={"name": "polar"})
        store = self._store()
        self.assertEqual([r.run_id for r in store.discover()], ["run1", "run2"])
        self.assertEqual([r.run_id for r in store.select(tag="baseline")], ["run1"])
        self.assertEqual([r.run_id for r in store.select(model="polar")], ["run2"])
        self.assertEqual([r.run_id for r in store.select(stage="stage1")], ["run1"])

    def test_unregistered_artifacts_are_not_discovered(self):
        write_prediction(self.root / "var/old/fold_0/validation/EMIDEC_P001.nii.gz")
        write_metrics(self.root / "var/results/old.json", "old", [("Case_P001", 0)])
        self.assertEqual(self._store().discover(), [])

    def test_incomplete_oof_reports_missing_cases(self):
        pred_root = self.root / "preds"
        write_prediction(pred_root / "fold_0/validation/EMIDEC_P001.nii.gz")
        write_manifest(pred_root, "partial",
                       data={"splits_path": str(self.splits)},
                       paths={"predictions": str(pred_root)})
        run = self._store().get("partial")
        summary = run.status_summary()
        self.assertFalse(run.is_complete())
        self.assertEqual(summary["prediction"]["missing_cases"], ["Case_P002"])
        self.assertEqual(summary["prediction"]["per_fold"][0]["present"], 1)
        self.assertEqual(summary["prediction"]["per_fold"][1]["present"], 0)
        self.assertEqual([r.run_id for r in self._store().select(complete=False)],
                         ["partial"])

    def test_status_summary_flags_split_drift(self):
        pred_root = self.root / "preds"
        write_prediction(pred_root / "fold_0/validation/EMIDEC_P001.nii.gz")
        write_manifest(pred_root, "drift",
                       data={"splits_path": str(self.splits),
                             "splits_sha256": "0" * 64},
                       paths={"predictions": str(pred_root)})
        self.assertFalse(self._store().get("drift").status_summary()["split_hash_match"])

    def test_load_prediction(self):
        pred_root = self.root / "preds"
        write_prediction(pred_root / "fold_0/validation/EMIDEC_P001.nii.gz")
        write_manifest(pred_root, "r",
                       data={"splits_path": str(self.splits)},
                       paths={"predictions": str(pred_root)})
        run = self._store().get("r")
        arr = run.load_pred("Case_P001")
        self.assertEqual(arr.shape, (2, 2, 1))

    def test_nested_get(self):
        manifest = RunManifest.from_dict({
            "run_id": "r",
            "model": {"name": "unet", "params": {"in_channels": 3}},
            "data": {"splits_sha256": "abc"},
        })
        self.assertEqual(manifest.get("model.params.in_channels"), 3)
        self.assertEqual(manifest.get("data.splits_sha256"), "abc")
        self.assertIsNone(manifest.get("model.params.missing"))
        self.assertEqual(manifest.get("model.params.missing", "fallback"), "fallback")
        self.assertIsNone(manifest.get("model.missing.deeper"))

    def test_roundtrip_preserves_unknown_keys(self):
        manifest = RunManifest.from_dict({"run_id": "r", "custom": {"a": 1}})
        self.assertEqual(manifest.to_dict()["custom"], {"a": 1})


class ResultFilterTests(unittest.TestCase):
    def test_pred_empty_requires_zero_not_unknown(self):
        from types import SimpleNamespace

        from scripts.runs import _row_matches

        args = SimpleNamespace(class_name=None, group=None, fold=None,
                               gt_present=False, pred_empty=True,
                               fp_min=None, fn_min=None)
        self.assertTrue(_row_matches({"pred_voxels": 0}, args))
        self.assertFalse(_row_matches({"pred_voxels": None}, args))
        self.assertFalse(_row_matches({"pred_voxels": 3}, args))


if __name__ == "__main__":
    unittest.main()
