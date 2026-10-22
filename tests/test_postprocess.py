"""Ordered post-processing metadata and directory application."""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from src.datautils.postprocess import PostProcessor, apply_to_dir


class PostProcessTests(unittest.TestCase):
    def test_ops_order_and_parameters(self):
        ops = PostProcessor(min_scar_voxels=50, min_mvo_voxels=10,
                            scar_grow_px=1).ops()
        self.assertEqual([op["name"] for op in ops],
                         ["gate_small_scar", "grow_scar",
                          "gate_small_mvo", "enforce_containment"])
        self.assertEqual(ops[0]["params"]["min_voxels"], 50)
        self.assertEqual(ops[1]["params"]["px"], 1)
        self.assertEqual(ops[2]["params"]["min_voxels"], 10)

    def test_apply_to_dir_preserves_layout_and_gates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "preds/fold_0/validation"
            src.mkdir(parents=True)
            labels = np.zeros((5, 5, 1), dtype=np.uint8)
            labels[0, 0] = 3      # 4 scar voxels: below the gate
            labels[1, 1] = 3
            labels[2, 2] = 3
            labels[3, 3] = 3
            labels[4, 4] = 4      # MVO without surviving scar
            nib.save(nib.Nifti1Image(labels, np.eye(4)),
                     str(src / "EMIDEC_P001.nii.gz"))

            out = root / "pp"
            written = apply_to_dir(root / "preds", out,
                                   PostProcessor(min_scar_voxels=50,
                                                 min_mvo_voxels=10))
            self.assertEqual(len(written), 1)
            dst = out / "fold_0/validation/EMIDEC_P001.nii.gz"
            self.assertTrue(dst.exists())
            result = np.asanyarray(nib.load(str(dst)).dataobj)
            self.assertEqual(int((result == 3).sum()), 0)
            self.assertEqual(int((result == 4).sum()), 0)


class PostProcessManifestTests(unittest.TestCase):
    def test_parent_link_and_ops_are_recorded(self):
        import json

        from scripts.apply_postprocess import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "stage1" / "fold_0" / "validation"
            src.mkdir(parents=True)
            nib.save(nib.Nifti1Image(np.zeros((3, 3, 1), np.uint8), np.eye(4)),
                     str(src / "EMIDEC_P001.nii.gz"))
            splits = root / "splits.json"
            splits.write_text(json.dumps([
                {"train": [], "val": ["EMIDEC_P001"]}]))
            out = root / "stage1_pp"
            rc = main(["--pred-dir", str(root / "stage1"),
                       "--out-dir", str(out), "--run-id", "stage1_pp",
                       "--splits", str(splits), "--scar-grow-px", "0"])
            self.assertEqual(rc, 0)
            manifest = json.loads((out / "run.json").read_text())
            self.assertEqual(manifest["parent_run"], "stage1")
            self.assertEqual(manifest["paths"]["predictions"], str(out.resolve()))
            self.assertEqual(
                [op["name"] for op in manifest["postprocess"]["ops"]],
                ["gate_small_scar", "grow_scar", "gate_small_mvo",
                 "enforce_containment"])
            self.assertIsNotNone(manifest["data"]["splits_sha256"])


if __name__ == "__main__":
    unittest.main()
