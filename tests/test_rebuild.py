import json
import tempfile
import unittest
from pathlib import Path

from src.datautils.splits import CVSplits
from src.rebuild import (
    _base_name,
    _postprocess,
    build_plan,
    manifest_fields,
    write_manifests,
)
from src.results import is_evaluation_report
from src.runs import RunStore


def _touch_prediction(root: Path, name: str) -> Path:
    pred = root / "var" / name / "fold_0" / "validation"
    pred.mkdir(parents=True, exist_ok=True)
    (pred / "EMIDEC_P001.nii.gz").write_bytes(b"")
    return root / "var" / name


class RebuildHelpersTest(unittest.TestCase):
    def test_is_evaluation_report_rejects_non_mapping(self):
        self.assertFalse(is_evaluation_report([]))
        self.assertFalse(is_evaluation_report(None))
        self.assertFalse(is_evaluation_report("metrics"))
        self.assertTrue(is_evaluation_report({"per_case": []}))

    def test_base_name_and_postprocess(self):
        self.assertEqual(_base_name("nnUNet_validation_501_pp_grow1"),
                         "nnUNet_validation_501")
        self.assertEqual(_base_name("nnUNet_validation_500resenc_pp"),
                         "nnUNet_validation_500resenc")
        self.assertEqual(_base_name("polar_preds_add_t0.7"),
                         "polar_preds_add_t0.7")
        self.assertEqual(_postprocess("nnUNet_validation_501"), {})
        grow = _postprocess("nnUNet_validation_501_pp_grow1")
        self.assertEqual(grow["scar_grow_px"], 1)
        self.assertEqual(grow["ops"][1]["name"], "grow_scar")
        gated = _postprocess("nnUNet_validation_501_pp")
        self.assertEqual(gated["scar_grow_px"], 0)


class RebuildPlanTest(unittest.TestCase):
    def test_stage1_and_polar_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch_prediction(root, "nnUNet_validation_501_pp")
            _touch_prediction(root, "polar_preds_add_t0.7")
            results = root / "var/results"
            results.mkdir(parents=True)
            (results / "nnunet_501_cv_metrics_pp.json").write_text(
                json.dumps({"per_case": []}))
            (results / "polar_cv_metrics_add_t0.7.json").write_text(
                json.dumps({"per_case": []}))
            splits_path = root / "var/nnUNet_validation/splits_final.json"
            splits_path.parent.mkdir(parents=True)
            splits_path.write_text(json.dumps(
                [{"train": [], "val": ["EMIDEC_P001"]}]))
            splits = CVSplits.load(splits_path)

            plan = build_plan(root)
            by_id = {run.run_id: run for run in plan}
            self.assertEqual(set(by_id), {"nnUNet_validation_501_pp",
                                          "polar_preds_add_t0.7"})

            stage1 = manifest_fields(by_id["nnUNet_validation_501_pp"],
                                     splits, root)
            self.assertEqual(stage1["stage"], "stage1")
            self.assertEqual(stage1["parent_run"], "nnUNet_validation_501")
            self.assertEqual(stage1["data"]["label_scheme"], "regions")
            self.assertEqual(stage1["postprocess"]["scar_grow_px"], 0)
            self.assertEqual(stage1["paths"]["metrics"],
                             "var/results/nnunet_501_cv_metrics_pp.json")

            polar = manifest_fields(by_id["polar_preds_add_t0.7"], splits, root)
            self.assertEqual(polar["stage"], "stage2")
            self.assertEqual(polar["source_run"], "nnUNet_validation_501_pp")
            self.assertEqual(polar["merge"]["mode"], "additive")
            self.assertEqual(polar["merge"]["prob_thresh"], 0.7)
            self.assertEqual(polar["merge"]["dilation"], 0)
            self.assertFalse(polar["model"]["params"]["use_radius"])

    def test_write_manifests_links_predictions_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch_prediction(root, "nnUNet_validation_501_pp")
            results = root / "var/results"
            results.mkdir(parents=True)
            (results / "nnunet_501_cv_metrics_pp.json").write_text(
                json.dumps({"per_case": []}))
            splits_path = root / "var/nnUNet_validation/splits_final.json"
            splits_path.parent.mkdir(parents=True)
            splits_path.write_text(json.dumps(
                [{"train": [], "val": ["EMIDEC_P001"]}]))

            store = RunStore(repo_root=root)
            plan = build_plan(root)
            result = write_manifests(plan, store, root)
            self.assertEqual(result["written"], ["nnUNet_validation_501_pp"])
            manifest_path = (root / "var/runs/nnUNet_validation_501_pp/"
                             "run.json")
            self.assertTrue(manifest_path.exists())

            runs = store.discover(refresh=True)
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertTrue(run.manifest.get("provenance.legacy"))
            self.assertTrue(run.manifest.get("provenance.reconstructed"))
            self.assertEqual(run.pred_dir, root / "var/nnUNet_validation_501_pp")
            self.assertEqual(run.metrics_path,
                             root / "var/results/nnunet_501_cv_metrics_pp.json")


if __name__ == "__main__":
    unittest.main()
