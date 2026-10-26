"""Canonical layout, immutable splits, writers, and held-out verification."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.layout import (MARKER, artifact_id, canonical_catalog, read_json,
                        split_identity, verify_outputs, write_json, write_output)
from src.runs import RunStore


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        cases = [f"EMIDEC_P{i:03}" for i in range(100)]
        self.folds = [{"train": [c for i, c in enumerate(cases) if i % 5 != f],
                       "val": [c for i, c in enumerate(cases) if i % 5 == f]} for f in range(5)]
        self.split = self.repo / "data/splits/input.json"
        write_json(self.split, self.folds)
        self.digest = split_identity(self.split)[0]
        canonical = self.split.with_name(f"emidec-{self.digest}.json")
        self.split.rename(canonical)
        self.split = canonical
        write_json(self.repo / MARKER, {"default_splits": f"data://splits/{canonical.name}"})
        write_json(self.repo / "config/roots.json", {"profile": "local", "profiles": {
            "local": {"data": "data", "nnunet": "data/nnunet", "experiments": "experiments"}}})
        self.registry = {"outputs": {}}
        for name, exp in (("nnUNet_validation", "nnUNet_validation"),
                          ("nnUNet_validation_pp", "nnUNet_validation"),
                          ("503_best_pp", "cascade_503_resenc")):
            base = self.repo / "experiments" / exp / "outputs" / name
            for fold, split in enumerate(self.folds):
                for case in split["val"]:
                    path = base / f"predictions/fold_{fold}/validation/{case}.nii.gz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"opaque prediction")
            write_json(base / "metrics.json", {"run_id": name, "results": []})
            record = {"run_id": name, "stage": "stage2" if name == "503_best_pp" else "stage1",
                      "model": {"name": "nnunet"}, "inference": {"checkpoint": "best"},
                      "paths": {"predictions": f"experiments://{exp}/outputs/{name}/predictions",
                                "metrics": f"experiments://{exp}/outputs/{name}/metrics.json"},
                      "data": {"splits_path": f"data://splits/{self.split.name}", "splits_sha256": self.digest}}
            write_json(base / "output.json", record)
            self.registry["outputs"][name] = record
        for exp, names, default in (("nnUNet_validation", ["nnUNet_validation", "nnUNet_validation_pp"], "nnUNet_validation_pp"),
                                    ("cascade_503_resenc", ["503_best_pp"], "503_best_pp")):
            write_json(self.repo / "experiments" / exp / "experiment.json", {
                "experiment_id": exp, "outputs": names, "default_output": default})

    def test_verifier_preserves_original_splits_and_metadata(self):
        before = {str(p): p.read_bytes() for p in self.repo.rglob("*.json")}
        self.assertEqual(verify_outputs(self.repo), (2, 3))
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.repo.rglob("*.json")})
        self.assertEqual(split_identity(self.split)[0], self.digest)

    def test_discovery_and_writing_need_no_migration_marker_or_machine_config(self):
        (self.repo / MARKER).unlink()
        (self.repo / "config/roots.json").unlink()
        store = RunStore(repo_root=self.repo)
        self.assertEqual(len(store.discover()), 2)
        store.register("503_best_pp", description="Canonical update")
        self.assertEqual(store.get("503_best_pp").description, "Canonical update")
        self.assertEqual(verify_outputs(self.repo), (2, 3))
        self.assertFalse((self.repo / "var").exists())

    def test_verifier_rejects_missing_probabilities(self):
        path = self.repo / "experiments/cascade_503_resenc/outputs/503_best_pp/output.json"
        output = read_json(path)
        output["paths"]["probabilities"] = "nnunet://nnUNet_results/missing_probabilities"
        write_json(path, output)
        with self.assertRaisesRegex(ValueError, "Missing OOF probability"):
            verify_outputs(self.repo)

    def test_verifier_rejects_recorded_split_drift(self):
        path = self.repo / "experiments/cascade_503_resenc/outputs/503_best_pp/output.json"
        output = read_json(path)
        output["data"]["splits_sha256"] = "0" * 64
        write_json(path, output)
        with self.assertRaisesRegex(ValueError, "Split hash mismatch"):
            verify_outputs(self.repo)



    def test_discovery_and_registration_use_canonical_metadata(self):
        store = RunStore(repo_root=self.repo)
        self.assertEqual(len(store.discover()), 2)
        self.assertEqual(store.get("503_best_pp").manifest.stage, "stage2")
        store.register("503_best_pp", description="Updated directly", stage="stage2")
        canonical = read_json(self.repo / "experiments/cascade_503_resenc/outputs/503_best_pp/output.json")
        self.assertEqual(canonical["description"], "Updated directly")
        self.assertEqual(canonical["data"]["splits_sha256"], self.digest)
        self.assertFalse((self.repo / "var").exists())



    def test_canonical_experiment_identity_must_match_directory(self):
        path = self.repo / "experiments/cascade_503_resenc/experiment.json"
        experiment = read_json(path)
        experiment["experiment_id"] = "another_experiment"
        write_json(path, experiment)
        with self.assertRaisesRegex(ValueError, "Experiment ID does not match"):
            canonical_catalog(self.repo)

    def test_duplicate_canonical_output_entries_are_rejected(self):
        path = self.repo / "experiments/cascade_503_resenc/experiment.json"
        experiment = read_json(path)
        experiment["outputs"].append("503_best_pp")
        write_json(path, experiment)
        with self.assertRaisesRegex(ValueError, "Duplicate output IDs"):
            canonical_catalog(self.repo)


    def test_missing_oof_case_is_rejected(self):
        path = next((self.repo / "experiments/cascade_503_resenc/outputs/503_best_pp/predictions").glob("fold_*/validation/*.nii.gz"))
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing OOF prediction"):
            verify_outputs(self.repo)
        self.assertFalse(self.split.parent.is_symlink())



    def test_split_leakage_is_rejected(self):
        folds = json.loads(json.dumps(self.folds))
        folds[0]["train"].append(folds[0]["val"][0])
        write_json(self.split, folds)
        with self.assertRaisesRegex(ValueError, "leakage"):
            split_identity(self.split)



    def test_output_identity_does_not_use_predictions_directory_name(self):
        path = self.repo / "experiments/cascade_503_resenc/outputs/503_best_pp/predictions"
        self.assertEqual(artifact_id(path, self.repo), "503_best_pp")

    def test_config_uses_canonical_roots_in_a_fresh_python_process(self):
        source = Path(__file__).resolve().parents[1]
        for name in ("src/config.py", "src/layout.py", "src/experiment_registry.py"):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((source / name).read_bytes())
        env = dict(os.environ)
        for name in ("EMIDEC_GT_ROOT", "NNUNET_PRED_DIR", "NNUNET_SPLITS_PATH", "NNUNET_CV_OUT", "RADDI_PROFILE"):
            env.pop(name, None)
        env["PYTHONPATH"] = str(self.repo)
        result = subprocess.run([sys.executable, "-c",
            "import json; from src.config import DEFAULT_SPLITS_PATH, DEFAULT_GT_ROOT, DEFAULT_NNUNET_RAW; "
            "print(json.dumps([DEFAULT_SPLITS_PATH, DEFAULT_GT_ROOT, DEFAULT_NNUNET_RAW]))"],
            cwd=self.repo, env=env, text=True, capture_output=True, check=True)
        split, gt, raw = json.loads(result.stdout)
        self.assertEqual(Path(split), self.repo / "data/splits" / f"emidec-{self.digest}.json")
        self.assertEqual(Path(gt), self.repo / "data/source/emidec/train")
        self.assertEqual(Path(raw), self.repo / "data/nnunet/nnUNet_raw")

    def test_versioned_splits_cannot_be_overwritten_with_new_folds(self):
        spec = importlib.util.spec_from_file_location("layout_split_tests",
            Path(__file__).resolve().parents[1] / "src/datautils/splits.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["layout_split_tests"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop("layout_split_tests", None)
        path = self.repo / "data/splits" / f"emidec-{self.digest}.json"
        write_json(path, self.folds)
        splitter = module.StratifiedFolds.__new__(module.StratifiedFolds)
        splitter.path = str(path)
        splitter.folds = json.loads(json.dumps(self.folds))
        self.assertEqual(splitter.save(), str(path))
        splitter.folds.reverse()
        with self.assertRaisesRegex(ValueError, "immutable"):
            splitter.save()
        self.assertEqual(read_json(path), self.folds)
        wrong = path.with_name("emidec-" + "0" * 64 + ".json")
        write_json(wrong, self.folds)
        with self.assertRaisesRegex(ValueError, "filename"):
            module.CVSplits.load(wrong)

    def test_training_metadata_does_not_add_a_predictions_named_output(self):
        checkpoints = self.repo / "experiments/new_polar/checkpoints"
        checkpoints.mkdir(parents=True)
        manifest = {"run_id": "new_polar__training", "stage": "stage2", "model": {"name": "polar_unet"},
                    "data": {"splits_path": str(self.split), "splits_sha256": self.digest},
                    "paths": {"checkpoints": str(checkpoints)}}
        write_output(self.repo, manifest)
        self.assertTrue((self.repo / "experiments/new_polar/training.json").exists())
        self.assertEqual(canonical_catalog(self.repo)["outputs"].keys(), self.registry["outputs"].keys())


if __name__ == "__main__":
    unittest.main()
