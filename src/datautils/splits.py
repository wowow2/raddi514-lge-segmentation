"""Stratified k-fold splits for EMIDEC, persisted for reproducibility."""

import json
import os

from sklearn.model_selection import StratifiedKFold

DEFAULT_SPLITS_PATH = "splits/emidec_5fold.json"


class StratifiedFolds:
    """Stratified folds over cases (group + scar + MVO presence).

    Usage:
        folds = StratifiedFolds(cases).build()   # list of {"train","val"} id dicts
        folds.save()                             # -> splits/emidec_5fold.json
        train, val = folds.split_dataset(ds, fold=0)
    """

    def __init__(self, cases, n_folds: int = 5, seed: int = 42,
                 path: str = DEFAULT_SPLITS_PATH):
        self.cases = list(cases)
        self.n_folds = n_folds
        self.seed = seed
        self.path = path
        self.folds: list[dict] | None = None

    @staticmethod
    def _stratum(case) -> str:
        return f"{case.group}|scar={case.has_scar}|mvo={case.has_mvo}"

    def build(self) -> list[dict]:
        """Build the folds; also stored on ``self.folds``."""
        if len(self.cases) < self.n_folds:
            raise ValueError(
                f"Need >= {self.n_folds} cases, got {len(self.cases)}")
        strata = [self._stratum(c) for c in self.cases]
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=self.seed)
        self.folds = [
            {"train": [self.cases[i].case_id for i in tr],
             "val": [self.cases[i].case_id for i in va]}
            for tr, va in skf.split(self.cases, strata)
        ]
        return self.folds

    def save(self, path: str | None = None) -> str:
        """Persist folds to JSON. Returns the path written."""
        path = path or self.path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.folds if self.folds is not None else self.build(),
                      f, indent=2)
        return path

    @classmethod
    def load(cls, path: str = DEFAULT_SPLITS_PATH) -> list[dict]:
        """Load persisted folds from JSON."""
        with open(path) as f:
            return json.load(f)

    def split_dataset(self, dataset, fold: int,
                      folds: list[dict] | None = None) -> tuple[list, list]:
        """Return (train_cases, val_cases) for `fold` as EMIDECCase lists."""
        folds = folds if folds is not None else (
            self.folds if self.folds is not None else self.load(self.path))
        by_id = {c.case_id: c for c in dataset.cases}
        return ([by_id[cid] for cid in folds[fold]["train"]],
                [by_id[cid] for cid in folds[fold]["val"]])

    def __repr__(self) -> str:
        return (f"StratifiedFolds(n_cases={len(self.cases)}, "
                f"n_folds={self.n_folds}, seed={self.seed})")
