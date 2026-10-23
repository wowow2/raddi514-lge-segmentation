import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from src.cascade import ExportConfig, export_dataset


def _write_fake_case(root: Path) -> None:
    case_dir = root / "var/emidec_dataset/train/Case_P001"
    (case_dir / "Images").mkdir(parents=True)
    (case_dir / "Contours").mkdir()
    image = np.zeros((64, 64, 3), dtype=np.float32)
    image[30:50, 30:50, :] = 100.0
    gt = np.zeros((64, 64, 3), dtype=np.uint8)
    gt[40:44, 40:44, :] = 2
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(image, affine),
             str(case_dir / "Images/Case_P001.nii.gz"))
    nib.save(nib.Nifti1Image(gt, affine),
             str(case_dir / "Contours/Case_P001.nii.gz"))


def _write_fake_probs(root: Path) -> Path:
    prob_dir = root / "var/nnUNet_results/Dataset500_EMIDEC/probs_fake"
    fold_dir = prob_dir / "fold_0/validation"
    fold_dir.mkdir(parents=True)
    probs = np.zeros((5, 3, 64, 64), dtype=np.float32)
    probs[0] = 1.0
    probs[0, :, 30:50, 30:50] = 0.0
    probs[2, :, 30:50, 30:50] = 1.0
    np.savez_compressed(fold_dir / "EMIDEC_P001.npz", probabilities=probs)
    return prob_dir


class CascadeExportTest(unittest.TestCase):
    def test_no_probs_still_uses_stage1_for_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_case(root)
            prob_dir = _write_fake_probs(root)
            splits_path = root / "var/nnUNet_validation/splits_final.json"
            splits_path.parent.mkdir(parents=True)
            splits_path.write_text(json.dumps(
                [{"train": [], "val": ["EMIDEC_P001"]}]))

            summary = export_dataset(ExportConfig(
                out_dir=root / "raw",
                task_name="Dataset999_CascadeTest",
                prob_dir=prob_dir,
                splits_path=splits_path,
                gt_root=root / "var/emidec_dataset/train",
                pad=2,
                min_size=16,
                multiple=8,
                include_probs=False,
                overwrite=True,
                case_ids=("EMIDEC_P001",),
            ))

            self.assertEqual(summary.n_cases, 1)
            case = summary.cases[0]
            self.assertEqual(case.case_id, "Case_P001")
            self.assertEqual(case.gt_coverage, 1.0)
            self.assertLessEqual(case.bbox["x0"], 30)
            self.assertEqual(len(list(
                (root / "raw/Dataset999_CascadeTest/imagesTr").glob("*.nii.gz"))),
                1)

            dataset_json = json.loads(
                (root / "raw/Dataset999_CascadeTest/dataset.json").read_text())
            self.assertEqual(dataset_json["channel_names"], {"0": "LGE"})
            self.assertEqual(dataset_json["numTraining"], 1)
            self.assertTrue(
                (root / "raw/Dataset999_CascadeTest/labelsTr/"
                 "EMIDEC_P001.nii.gz").exists())
            self.assertTrue(
                (root / "raw/Dataset999_CascadeTest/"
                 "cascade_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
