"""Polar prediction and evaluation share model lineage and output metadata."""
import tempfile
import unittest
from pathlib import Path

from src.datautils.postprocess import PostProcessor
from src.datautils.splits import CVSplits
from src.layout import write_json
from src.polar.predict import prediction_manifest


class PolarMetadataTests(unittest.TestCase):
    def test_raw_and_gated_metadata_keep_same_lineage_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            write_json(models / "run.json", {
                "run_id": "polar_training", "description": "trained model",
                "tags": ["polar"], "model": {"name": "inherited_model"},
                "training": {"seed": 42}})
            splits = CVSplits.from_list([
                {"train": ["EMIDEC_P002"], "val": ["EMIDEC_P001"]},
                {"train": ["EMIDEC_P001"], "val": ["EMIDEC_P002"]}])
            splits.path = str(root / "splits.json")
            result = {"config": {"model": {"name": "checkpoint_model"}}, "R": 64,
                      "Theta": 256, "prediction": {"n_present": 2}}
            inference = {"mode": "erase", "dilation": 0, "tau": None, "device": "cpu"}
            raw = prediction_manifest(result, models, root / "raw", root / "stage1",
                                      splits, root / "gt", inference=inference)
            gate = PostProcessor()
            gated = prediction_manifest(result, models, root / "gated", root / "stage1",
                                        splits, root / "gt", inference=inference, gate=gate)
            self.assertEqual(raw.data, gated.data)
            self.assertEqual(raw.data["splits_sha256"], splits.sha256)
            self.assertEqual(raw.data["n_cases"], 2)
            self.assertEqual(raw.parent_run, "polar_training")
            self.assertEqual(raw.source_run, "stage1")
            self.assertEqual(raw.training, {"seed": 42})
            self.assertEqual(raw.model, {"name": "checkpoint_model"})
            self.assertEqual(raw.inference["checkpoint"], "best")
            self.assertEqual(raw.postprocess, {"ops": []})
            self.assertEqual(gated.postprocess["ops"], gate.ops())
            self.assertEqual(gated.postprocess["min_scar_voxels"], gate.min_scar_voxels)
            self.assertNotEqual(raw.paths["predictions"], gated.paths["predictions"])
            self.assertNotIn("metrics", raw.paths)


if __name__ == "__main__":
    unittest.main()
