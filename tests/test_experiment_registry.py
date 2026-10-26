"""Configured roots and canonical experiment/output grouping."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiment_registry import ArtifactRoots, catalog_views
from src.layout import write_json
from src.runs import RunStore


class RegistrationTests(unittest.TestCase):
    def test_paths_profiles_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = {"profile": "local", "profiles": {
                "local": {"data": "data", "nnunet": "data/nnunet"},
                "fir": {"data": "/scratch/data", "nnunet": "/scratch/data/nnunet"}},
                "aliases": {"/scratch/data/nnunet/": "nnunet://"}}
            with patch.dict("os.environ", {"RADDI_PROFILE": "local"}):
                roots = ArtifactRoots(repo, config)
            self.assertEqual(roots.resolve("data://source"), repo / "data/source")
            self.assertEqual(roots.portable(str(repo / "data/source")), "data://source")
            self.assertEqual(roots.portable("/scratch/data/nnunet/x"), "nnunet://x")
            with self.assertRaisesRegex(ValueError, "escapes"):
                roots.resolve("data://../../outside")
            with patch.dict("os.environ", {"RADDI_PROFILE": "fir"}):
                self.assertEqual(ArtifactRoots(repo, config).resolve("nnunet://x"), Path("/scratch/data/nnunet/x"))

    def test_default_roots_work_without_machine_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            roots = ArtifactRoots(repo)
            self.assertEqual(roots.resolve("experiments://polar"), repo / "experiments/polar")
            self.assertEqual(roots.resolve("nnunet://nnUNet_raw"), repo / "data/nnunet/nnUNet_raw")

    def test_grouping_preserves_final_and_best(self):
        catalog = {"outputs": {
            "final": {"run_id": "final", "inference": {"checkpoint": "final"}},
            "best": {"run_id": "best", "inference": {"checkpoint": "best"}}},
            "membership": {"final": "cascade", "best": "cascade"},
            "defaults": {"cascade": "best"}}
        experiments, outputs = catalog_views(catalog)
        self.assertEqual(experiments["cascade"]["default_output"], "best")
        self.assertEqual(experiments["cascade"]["outputs"], ["best", "final"])
        self.assertEqual(outputs["final"]["inference"]["checkpoint"], "final")
        outputs["best"]["inference"]["checkpoint"] = "changed"
        self.assertEqual(catalog["outputs"]["best"]["inference"]["checkpoint"], "best")

    def test_final_checkpoint_never_falls_back_to_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            exp = repo / "experiments/cascade"
            weights = repo / "weights/fold_0"
            weights.mkdir(parents=True)
            (weights / "checkpoint_best.pth").touch()
            write_json(exp / "experiment.json", {
                "experiment_id": "cascade", "outputs": ["final"], "default_output": "final"})
            write_json(exp / "outputs/final/output.json", {
                "run_id": "final", "inference": {"checkpoint": "final"},
                "paths": {"checkpoints": str(weights.parent)}})
            run = RunStore(repo_root=repo).get("final")
            self.assertIsNone(run.checkpoint())
            (weights / "checkpoint_final.pth").touch()
            self.assertEqual(run.checkpoint().name, "checkpoint_final.pth")


if __name__ == "__main__":
    unittest.main()
