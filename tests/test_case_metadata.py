"""Case-level metadata is stored separately and joined by case id."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from src.datautils.case_metadata import (
    case_metadata,
    load_case_metadata,
    save_case_metadata,
)
from src.datautils.emidec_loader import EMIDECCase


def make_case(root: Path, case_id: str, with_gt: bool = True) -> EMIDECCase:
    case_dir = root / case_id
    (case_dir / "Images").mkdir(parents=True)
    image = np.arange(4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2)
    affine = np.diag([1.5, 1.5, 10.0, 1.0])
    nib.save(nib.Nifti1Image(image, affine), str(case_dir / "Images" / f"{case_id}.nii.gz"))
    if with_gt:
        (case_dir / "Contours").mkdir()
        gt = np.zeros((4, 5, 2), dtype=np.uint8)
        gt[0, 0, 0] = 1
        gt[1, 1, 0] = 3
        nib.save(nib.Nifti1Image(gt, affine),
                 str(case_dir / "Contours" / f"{case_id}.nii.gz"))
    return EMIDECCase(str(case_dir))


class CaseMetadataTests(unittest.TestCase):
    def test_collect_and_join(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cases = [make_case(root, "Case_P001"), make_case(root, "Case_N002")]
            dataset = SimpleNamespace(cases=cases)
            path = root / "cases.json"
            save_case_metadata(path, dataset)
            loaded = load_case_metadata(path)
            self.assertEqual(set(loaded), {"Case_P001", "Case_N002"})
            self.assertEqual(loaded["Case_P001"]["n_slices"], 2)
            self.assertEqual(loaded["Case_P001"]["shape"], [4, 5, 2])
            self.assertEqual(loaded["Case_P001"]["gt_voxels"]["3"], 1)
            self.assertEqual(loaded["Case_N002"]["spacing"], [1.5, 1.5, 10.0])
            payload = json.loads(path.read_text())
            self.assertEqual(payload["schema_version"], 1)

    def test_blind_case_has_zero_counts(self):
        with tempfile.TemporaryDirectory() as td:
            case = make_case(Path(td), "Case_101", with_gt=False)
            meta = case_metadata(case)
            self.assertFalse(meta["has_gt"])
            self.assertEqual(meta["gt_voxels"][1], 0)
            self.assertEqual(meta["case_id"], "Case_101")

    def test_does_not_retain_image_array(self):
        with tempfile.TemporaryDirectory() as td:
            case = make_case(Path(td), "Case_P003")
            case_metadata(case)
            self.assertIsNone(case._img)
            self.assertIsNone(case._gt)


if __name__ == "__main__":
    unittest.main()
