"""Split identity and validation."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.config import case_id_to_nnunet_id, nnunet_id_to_case_id
from src.datautils.splits import CVSplits, StratifiedFolds


class IdConversionTests(unittest.TestCase):
    def test_idempotent(self):
        for value in ("Case_P001", "EMIDEC_P001"):
            self.assertEqual(nnunet_id_to_case_id(value), "Case_P001")
            self.assertEqual(case_id_to_nnunet_id(value), "EMIDEC_P001")


class CVSplitsTests(unittest.TestCase):
    def _splits(self, folds):
        return CVSplits(folds)

    def test_accepts_both_id_forms_and_hashes_content(self):
        folds = [
            {"train": ["Case_N001"], "val": ["Case_P001"]},
            {"train": ["Case_P001"], "val": ["EMIDEC_N001"]},
        ]
        splits = self._splits(folds)
        self.assertEqual(splits.fold_of("Case_P001"), 0)
        self.assertEqual(splits.fold_of("EMIDEC_P001"), 0)
        self.assertEqual(splits.case_ids(), ["Case_P001", "Case_N001"])
        self.assertEqual(splits.sha256, self._splits(json.loads(json.dumps(folds))).sha256)

    def test_duplicate_validation_membership(self):
        splits = self._splits([
            {"train": [], "val": ["EMIDEC_P001"]},
            {"train": [], "val": ["EMIDEC_P001"]},
        ])
        errors = splits.validate().errors
        self.assertTrue(any("appears in validation" in e for e in errors))

    def test_train_val_overlap(self):
        splits = self._splits([
            {"train": ["EMIDEC_P001"], "val": ["EMIDEC_P001"]},
        ])
        self.assertTrue(any("both train and val" in e for e in splits.validate().errors))

    def test_duplicate_within_fold(self):
        splits = self._splits([
            {"train": [], "val": ["Case_P001", "EMIDEC_P001"]},
        ])
        self.assertTrue(any("duplicate ids" in e for e in splits.validate().errors))

    def test_missing_dataset_case(self):
        splits = self._splits([{"train": ["EMIDEC_P002"], "val": ["EMIDEC_P001"]}])
        result = splits.validate(case_ids=["Case_P001", "Case_P003"])
        self.assertTrue(any("never appear" in e for e in result.errors))

    def test_load_rejects_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "splits.json"
            path.write_text(json.dumps([{"train": [], "val": []}]))
            with self.assertRaises(ValueError):
                CVSplits.load(path)
            self.assertFalse(CVSplits.load(path, validate=False).validate().ok)


class StratifiedFoldsTests(unittest.TestCase):
    def test_writes_nnunet_ids_and_splits_both_forms(self):
        cases = [SimpleNamespace(case_id=f"Case_{i:03d}", group="Normal",
                                 has_scar=False, has_mvo=False)
                 for i in range(1, 7)]
        builder = StratifiedFolds(cases, n_folds=3, seed=0)
        folds = builder.build()
        for split in folds:
            for value in split["train"] + split["val"]:
                self.assertTrue(value.startswith("EMIDEC_"))
        dataset = SimpleNamespace(cases=cases)
        train, val = builder.split_dataset(dataset, 0, folds=folds)
        self.assertEqual(len(train) + len(val), 6)


if __name__ == "__main__":
    unittest.main()
