"""Run discovery, manifests, legacy fallback, and comparison guards."""

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

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
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, **fields}
    path = directory / "run.json"
    path.write_text(json.dumps(payload))
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

    def _store(self, roots, **kwargs):
        return RunStore(roots=roots, legacy_roots=kwargs.pop("legacy_roots", []),
                        repo_root=self.root, splits_path=self.splits, **kwargs)

    def test_discovery_filters_and_merge(self):
        a = self.root / "a"
        b = self.root / "b"
        write_manifest(a, "run1", stage="stage1", tags=["baseline"],
                       model={"name": "unet"},
                       data={"splits_path": str(self.splits), "splits_sha256": "abc"},
                       paths={"predictions": str(a)})
        write_manifest(b, "run1", stage="stage1", tags=["gated"],
                       paths={"metrics": str(b / "metrics.json")})
        store = self._store([a, b])
        runs = store.discover()
        self.assertEqual([r.run_id for r in runs], ["run1"])
        run = runs[0]
        self.assertEqual(run.tags, ["baseline", "gated"])
        self.assertEqual(run.manifest.get("model.name"), "unet")
        self.assertEqual(run.metrics_path, b / "metrics.json")

        write_manifest(self.root / "c", "run2", stage="stage2", model={"name": "polar"})
        store = self._store([a, b, self.root / "c"])
        self.assertEqual([r.run_id for r in store.select(tag="baseline")], ["run1"])
        self.assertEqual([r.run_id for r in store.select(model="polar")], ["run2"])
        self.assertEqual([r.run_id for r in store.select(stage="stage1")], ["run1"])

    def test_legacy_prediction_dir_is_read_only_and_marked_unknown(self):
        pred = self.root / "legacy" / "fold_0" / "validation"
        write_prediction(pred / "EMIDEC_P001.nii.gz")
        store = self._store([], legacy_roots=[self.root / "legacy"])
        run = store.get("legacy")
        self.assertTrue(run.manifest.legacy)
        self.assertIsNone(run.manifest.get("model.name"))
        self.assertEqual(run.status()["n_present"], 1)
        self.assertIsNone(run.status()["n_expected"])  # no split, coverage unknown
        self.assertIsNone(run.status()["complete"])
        self.assertFalse(run.status()["fully_tracked"])

    def test_legacy_metrics_file_discovered(self):
        path = self.root / "var/results/old_run.json"
        write_metrics(path, "old_run", [("Case_P001", 0), ("Case_P002", 1)])
        store = self._store([])
        run = store.get("old_run")
        self.assertTrue(run.manifest.legacy)
        table = run.load_metrics()
        self.assertEqual(len(table), 8)
        self.assertEqual(len(table.case_ids()), 2)

    def test_register_claims_legacy_prediction_dir(self):
        pred_root = self.root / "legacy2"
        write_prediction(pred_root / "fold_0/validation/EMIDEC_P001.nii.gz")
        store = self._store([], legacy_roots=[pred_root])
        store.register("registered", stage="stage1",
                       data={"splits_path": str(self.splits)},
                       paths={"predictions": str(pred_root)})
        runs = self._store([], legacy_roots=[pred_root]).discover()
        self.assertEqual([r.run_id for r in runs], ["registered"])
        self.assertEqual(runs[0].status()["n_present"], 1)

    def test_register_persists_and_merges(self):
        store = self._store([])
        store.register("manual", stage="stage1", tags=["x"],
                       paths={"predictions": str(self.root / "p")},
                       model={"name": "unet"})
        self.assertTrue((self.root / "var/runs/manual/run.json").exists())
        reloaded = self._store([])
        run = reloaded.get("manual")
        self.assertEqual(run.manifest.stage, "stage1")
        self.assertEqual(run.tags, ["x"])
        run.manifest.missing_fields()  # must not raise for partial metadata

    def test_status_reports_missing_cases(self):
        pred_root = self.root / "preds"
        write_prediction(pred_root / "fold_0/validation/EMIDEC_P001.nii.gz")
        write_manifest(pred_root, "partial",
                       data={"splits_path": str(self.splits)},
                       paths={"predictions": str(pred_root)})
        run = self._store([pred_root]).get("partial")
        status = run.status()
        self.assertFalse(status["complete"])
        self.assertEqual(status["missing_cases"], ["Case_P002"])
        self.assertEqual(status["prediction"]["per_fold"][0]["present"], 1)
        self.assertEqual(status["prediction"]["per_fold"][1]["present"], 0)

    def test_status_flags_split_drift(self):
        pred_root = self.root / "preds"
        write_prediction(pred_root / "fold_0/validation/EMIDEC_P001.nii.gz")
        write_manifest(pred_root, "drift",
                       data={"splits_path": str(self.splits),
                             "splits_sha256": "0" * 64},
                       paths={"predictions": str(pred_root)})
        status = self._store([pred_root]).get("drift").status()
        self.assertTrue(any("does not match" in e for e in status["split_errors"]))

    def test_load_prediction(self):
        pred_root = self.root / "preds"
        write_prediction(pred_root / "fold_0/validation/EMIDEC_P001.nii.gz")
        write_manifest(pred_root, "r",
                       data={"splits_path": str(self.splits)},
                       paths={"predictions": str(pred_root)})
        run = self._store([pred_root]).get("r")
        arr = run.load_pred("Case_P001")
        self.assertEqual(arr.shape, (2, 2, 1))

    def test_conflicting_manifests_are_flagged_and_compare_rejects(self):
        a = self.root / "a"
        b = self.root / "b"
        pred = self.root / "pred"
        write_prediction(pred / "fold_0/validation/EMIDEC_P001.nii.gz")
        write_manifest(a, "same", stage="stage1",
                       model={"name": "unet", "params": {"in_channels": 1}},
                       data={"splits_path": str(self.splits), "splits_sha256": "a" * 64},
                       paths={"predictions": str(pred)})
        write_manifest(b, "same", stage="stage1",
                       model={"name": "unet", "params": {"in_channels": 1}},
                       data={"splits_path": str(self.splits), "splits_sha256": "b" * 64},
                       paths={"predictions": str(pred)})
        write_metrics(a / "metrics.json", "same", [("Case_P001", 0)])
        store = self._store([a, b])
        run = store.get("same")
        conflicts = run.status()["manifest_conflicts"]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["field"], "data.splits_sha256")
        self.assertEqual(len(conflicts[0]["sources"]), 2)
        with self.assertRaises(ValueError):
            store.compare(["same"])

    def test_register_rejects_identity_conflict(self):
        pred = self.root / "pred"
        write_prediction(pred / "fold_0/validation/EMIDEC_P001.nii.gz")
        store = self._store([])
        store.register("reg", data={"splits_sha256": "a" * 64},
                       paths={"predictions": str(pred)})
        with self.assertRaises(ValueError):
            store.register("reg", data={"splits_sha256": "b" * 64})

    def test_polar_variant_directories_are_discovered(self):
        for name in ("polar_preds_raw", "polar_preds_pp", "polar_models_radius"):
            directory = self.root / "var" / name
            write_manifest(directory, name, stage="stage2",
                           paths={"predictions": str(directory)})
        store = RunStore(roots=[], legacy_roots=None, repo_root=self.root,
                         splits_path=self.splits)
        found = {run.run_id for run in store.discover()}
        self.assertTrue({"polar_preds_raw", "polar_preds_pp",
                         "polar_models_radius"} <= found)

    def test_parent_and_source_links_persist(self):
        store = self._store([])
        store.register("child", parent_run="parent", source_run="stage1",
                       stage="stage2", paths={"predictions": str(self.root / "p")})
        run = self._store([]).get("child")
        self.assertEqual(run.manifest.parent_run, "parent")
        self.assertEqual(run.manifest.source_run, "stage1")

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

    def test_compare_rejects_mixed_splits_and_missing_metrics(self):
        good = self.root / "good"
        other = self.root / "other"
        for directory, run_id, sha in ((good, "a", "a" * 64),
                                       (other, "b", "b" * 64)):
            write_metrics(directory / "metrics.json", run_id,
                          [("Case_P001", 0), ("Case_P002", 1)])
            write_manifest(directory, run_id,
                           data={"splits_path": str(self.splits), "splits_sha256": sha},
                           paths={"metrics": str(directory / "metrics.json")})
        store = self._store([good, other])
        with self.assertRaises(ValueError):
            store.compare(["a", "b"])

    def test_compare_requires_metrics(self):
        directory = self.root / "empty"
        write_manifest(directory, "empty", paths={"predictions": str(directory)})
        store = self._store([directory])
        with self.assertRaises(FileNotFoundError):
            store.compare(["empty"])


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


class ManifestTests(unittest.TestCase):
    def test_merge_prefers_known_values(self):
        primary = RunManifest(run_id="r", stage="stage2",
                              data={"splits_sha256": "abc", "gt_root": None},
                              paths={"predictions": "p"})
        secondary = RunManifest(run_id="r", stage="stage1", tags=["old"],
                                data={"splits_sha256": "other", "gt_root": "gt"},
                                training={"seed": 1},
                                paths={"metrics": "m"})
        merged = primary.merged_with(secondary)
        self.assertEqual(merged.stage, "stage2")
        self.assertEqual(merged.data["splits_sha256"], "abc")
        self.assertEqual(merged.data["gt_root"], "gt")
        self.assertEqual(merged.paths["metrics"], "m")
        self.assertEqual(merged.training["seed"], 1)
        self.assertEqual(merged.tags, ["old"])

    def test_unknown_fields_are_none_not_guessed(self):
        manifest = RunManifest.from_dict({"run_id": "r"})
        self.assertIsNone(manifest.get("model.name"))
        self.assertEqual(manifest.get("paths.predictions"), None)
        self.assertIn("paths.predictions", manifest.missing_fields())

    def test_roundtrip_preserves_unknown_keys(self):
        manifest = RunManifest.from_dict({"run_id": "r", "custom": {"a": 1}})
        self.assertEqual(manifest.to_dict()["custom"], {"a": 1})


if __name__ == "__main__":
    unittest.main()
