"""Cross-validation splits: creation, validation, and identity.

Two formats coexist in this project:

* ``splits_final.json`` written by nnU-Net (and used by every prediction
  directory) stores ``EMIDEC_*`` ids;
* :class:`StratifiedFolds` historically stored ``Case_*`` ids.

Both are accepted on read. :class:`CVSplits` normalises ids, validates the
fold structure, and exposes a content hash so runs can be compared by the
exact split they were trained/evaluated on.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field

from sklearn.model_selection import StratifiedKFold

from src.config import (
    DEFAULT_SPLITS_PATH,
    case_id_to_nnunet_id,
    nnunet_id_to_case_id,
)

@dataclass
class SplitValidation:
    """Result of validating a fold list."""

    n_folds: int = 0
    n_cases: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise ValueError("Invalid CV splits:\n  - " + "\n  - ".join(self.errors))

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_folds": self.n_folds,
            "n_cases": self.n_cases,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class CVSplits:
    """A validated, hashable list of ``{"train": [...], "val": [...]}`` folds.

    Usage::

        splits = CVSplits.load("var/nnUNet_validation/splits_final.json")
        fold = splits.fold_of("Case_P001")       # accepts either id form
        splits.sha256                            # identity for run comparison
    """

    def __init__(self, folds: list[dict], path: str | os.PathLike | None = None):
        self.folds = list(folds)
        self.path = str(path) if path is not None else None
        self._fold_of: dict[str, int] = {}

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | os.PathLike | None = None,
             validate: bool = True) -> "CVSplits":
        path = path or DEFAULT_SPLITS_PATH
        with open(path) as f:
            folds = json.load(f)
        splits = cls(folds, path=path)
        if validate:
            splits.validate().raise_if_invalid()
        return splits

    @classmethod
    def from_list(cls, folds: list[dict], validate: bool = True) -> "CVSplits":
        splits = cls(folds)
        if validate:
            splits.validate().raise_if_invalid()
        return splits

    # -------------------------------------------------------------- identity
    @property
    def sha256(self) -> str:
        """Hash of canonical fold contents (independent of file formatting)."""
        canonical = json.dumps(self.folds, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "n_folds": len(self.folds),
            "n_cases": len(self.case_ids()),
            "fold_sizes": {i: len(f.get("val", [])) for i, f in enumerate(self.folds)},
        }

    # --------------------------------------------------------------- queries
    def _ids(self, values) -> list[str]:
        return [nnunet_id_to_case_id(v) for v in values]

    def val_ids(self, fold: int) -> list[str]:
        return self._ids(self.folds[fold].get("val", []))

    def train_ids(self, fold: int) -> list[str]:
        return self._ids(self.folds[fold].get("train", []))

    def case_ids(self) -> list[str]:
        seen: list[str] = []
        for split in self.folds:
            for cid in self._ids(split.get("val", [])):
                if cid not in seen:
                    seen.append(cid)
        return seen

    def fold_of(self, case_id: str) -> int:
        """Validation fold for a case (accepts ``Case_*`` or ``EMIDEC_*``)."""
        if not self._fold_of:
            for fold, split in enumerate(self.folds):
                for cid in self._ids(split.get("val", [])):
                    self._fold_of.setdefault(cid, fold)
        cid = nnunet_id_to_case_id(case_id)
        if cid not in self._fold_of:
            raise KeyError(f"{cid} is not a validation case in these splits")
        return self._fold_of[cid]

    def fold_of_map(self) -> dict[str, int]:
        if not self._fold_of:
            for fold, split in enumerate(self.folds):
                for cid in self._ids(split.get("val", [])):
                    self._fold_of.setdefault(cid, fold)
        return dict(self._fold_of)

    # ------------------------------------------------------------ validation
    def validate(self, case_ids: list[str] | None = None,
                 expected_folds: int | None = None) -> SplitValidation:
        result = SplitValidation(n_folds=len(self.folds),
                                 n_cases=len(self.case_ids()))
        if not self.folds:
            result.errors.append("splits file contains no folds")
            return result
        if expected_folds is not None and len(self.folds) != expected_folds:
            result.errors.append(
                f"expected {expected_folds} folds, found {len(self.folds)}")

        seen_val: dict[str, int] = {}
        all_ids: set[str] = set()
        for fold, split in enumerate(self.folds):
            if not isinstance(split, dict):
                result.errors.append(f"fold {fold}: expected a dict")
                continue
            if not isinstance(split.get("train"), list) or not isinstance(split.get("val"), list):
                result.errors.append(f"fold {fold}: 'train' and 'val' must be lists")
                continue
            train = self._ids(split["train"])
            val = self._ids(split["val"])
            if not val:
                result.errors.append(f"fold {fold}: empty validation list")
            for name, ids in (("train", train), ("val", val)):
                if len(ids) != len(set(ids)):
                    dupes = sorted({cid for cid in ids if ids.count(cid) > 1})
                    result.errors.append(
                        f"fold {fold}: duplicate ids in {name}: {', '.join(dupes[:5])}")
            overlap = sorted(set(train) & set(val))
            if overlap:
                result.errors.append(
                    f"fold {fold}: {len(overlap)} case(s) in both train and val "
                    f"(e.g. {', '.join(overlap[:3])})")
            for cid in val:
                if cid in seen_val and seen_val[cid] != fold:
                    result.errors.append(
                        f"{cid} appears in validation of folds "
                        f"{seen_val[cid]} and {fold}")
                seen_val.setdefault(cid, fold)
            all_ids.update(train)
            all_ids.update(val)

        if case_ids is not None:
            expected = {nnunet_id_to_case_id(c) for c in case_ids}
            missing = sorted(expected - set(seen_val))
            extra = sorted(set(seen_val) - expected)
            if missing:
                result.errors.append(
                    f"{len(missing)} dataset case(s) never appear in a validation "
                    f"fold (e.g. {', '.join(missing[:5])})")
            if extra:
                result.warnings.append(
                    f"{len(extra)} validation case(s) are absent from the dataset "
                    f"(e.g. {', '.join(extra[:5])})")
            uncovered = sorted(expected - all_ids)
            if uncovered:
                result.warnings.append(
                    f"{len(uncovered)} case(s) missing from train and val lists "
                    f"(e.g. {', '.join(uncovered[:5])})")
        return result


class StratifiedFolds:
    """Stratified folds over cases (group + scar + MVO presence).

    .. note::
        Writes nnU-Net style ``EMIDEC_*`` ids so the result can be dropped in as
        ``splits_final.json``. :meth:`split_dataset` accepts either id form.
    """

    def __init__(self, cases, n_folds: int = 5, seed: int = 42,
                 path: str | None = None):
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
            {"train": [case_id_to_nnunet_id(self.cases[i].case_id) for i in tr],
             "val": [case_id_to_nnunet_id(self.cases[i].case_id) for i in va]}
            for tr, va in skf.split(self.cases, strata)
        ]
        return self.folds

    def save(self, path: str | None = None) -> str:
        """Persist folds to JSON. Returns the path written."""
        path = path or self.path
        if path is None:
            raise ValueError("StratifiedFolds has no output path; pass --path")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.folds if self.folds is not None else self.build(),
                      f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | None = None) -> list[dict]:
        """Load persisted folds from JSON."""
        with open(path) as f:
            return json.load(f)

    def split_dataset(self, dataset, fold: int,
                      folds: list[dict] | None = None) -> tuple[list, list]:
        """Return (train_cases, val_cases) for ``fold`` as case objects."""
        folds = folds if folds is not None else (
            self.folds if self.folds is not None else self.load(self.path))
        by_id = {nnunet_id_to_case_id(c.case_id): c for c in dataset.cases}
        return ([by_id[nnunet_id_to_case_id(cid)] for cid in folds[fold]["train"]],
                [by_id[nnunet_id_to_case_id(cid)] for cid in folds[fold]["val"]])

    def __repr__(self) -> str:
        return (f"StratifiedFolds(n_cases={len(self.cases)}, "
                f"n_folds={self.n_folds}, seed={self.seed})")
