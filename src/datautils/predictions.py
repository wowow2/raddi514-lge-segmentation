"""Shared lookup for out-of-fold prediction / probability files.

Every consumer (evaluation, plotting, viewer, polar anchors, probability
diagnostics) used to re-implement the same ``splits_final.json`` +
``fold_<k>/validation/<id>.nii.gz`` lookup. :class:`CVPredictionIndex` is the
single implementation.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    DEFAULT_SPLITS_PATH,
    case_id_to_nnunet_id,
    nnunet_id_to_case_id,
)
from src.datautils.splits import CVSplits


@dataclass(frozen=True)
class PredictionFile:
    case_id: str
    nnunet_id: str
    fold: int
    path: Path
    exists: bool


class CVPredictionIndex:
    """Resolve case -> prediction path for an out-of-fold directory.

    Args:
        pred_dir: directory containing ``fold_<k>/validation/``.
        splits: :class:`CVSplits`; loaded from ``splits_final.json`` beside
            ``pred_dir`` (or the default path) when omitted.
        suffix: ``.nii.gz`` for labels, ``.npz`` for probabilities.
        allow_scan: when no splits are available, fall back to scanning the
            fold directories. Fold assignment is then unknown for cases that
            appear in more than one fold.
    """

    def __init__(self, pred_dir: str | os.PathLike,
                 splits: CVSplits | None = None,
                 suffix: str = ".nii.gz",
                 allow_scan: bool = True,
                 splits_path: str | os.PathLike | None = None,
                 use_default_splits: bool = True):
        self.pred_dir = Path(pred_dir)
        self.suffix = suffix
        self.splits = (splits if splits is not None
                       else self._find_splits(splits_path, use_default_splits))
        self.splits_source = str(self.splits.path) if self.splits and self.splits.path else None
        self._paths: dict[str, PredictionFile] = {}
        self._unknown_folds: list[str] = []

        if self.splits is not None:
            for fold, split in enumerate(self.splits.folds):
                for raw in split.get("val", []):
                    cid = nnunet_id_to_case_id(raw)
                    nid = case_id_to_nnunet_id(raw)
                    path = self.pred_dir / f"fold_{fold}" / "validation" / f"{nid}{suffix}"
                    self._paths[cid] = PredictionFile(cid, nid, fold, path, path.exists())
        elif allow_scan:
            self._scan()

    def _find_splits(self, explicit: str | os.PathLike | None = None,
                     use_default: bool = True) -> CVSplits | None:
        candidates = [explicit] if explicit else []
        candidates.append(self.pred_dir / "splits_final.json")
        if use_default:
            candidates.append(Path(DEFAULT_SPLITS_PATH))
        for candidate in candidates:
            if candidate is None:
                continue
            candidate = Path(candidate)
            if candidate.exists():
                try:
                    return CVSplits.load(candidate)
                except (OSError, ValueError):
                    continue
        return None

    def _scan(self) -> None:
        for fold_dir in sorted(self.pred_dir.glob("fold_*")):
            try:
                fold = int(fold_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            for path in sorted((fold_dir / "validation").glob(f"*{self.suffix}")):
                nid = path.name[: -len(self.suffix)]
                cid = nnunet_id_to_case_id(nid)
                if cid in self._paths and self._paths[cid].fold != fold:
                    self._unknown_folds.append(cid)
                    continue
                self._paths[cid] = PredictionFile(cid, nid, fold, path, True)

    # --------------------------------------------------------------- queries
    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, case_id: str) -> bool:
        return nnunet_id_to_case_id(case_id) in self._paths

    def case_ids(self) -> list[str]:
        return sorted(self._paths)

    def items(self) -> dict[str, PredictionFile]:
        """Case id -> resolved :class:`PredictionFile`."""
        return dict(self._paths)

    def fold_of(self, case_id: str) -> int | None:
        ref = self._paths.get(nnunet_id_to_case_id(case_id))
        return None if ref is None else ref.fold

    def ref(self, case_id: str) -> PredictionFile:
        cid = nnunet_id_to_case_id(case_id)
        if cid not in self._paths:
            raise KeyError(
                f"{cid} has no expected output in {self.pred_dir} "
                f"(splits={'loaded' if self.splits else 'unavailable'})")
        return self._paths[cid]

    def path(self, case_id: str) -> Path:
        return self.ref(case_id).path

    def available(self) -> list[str]:
        return sorted(cid for cid, ref in self._paths.items() if ref.exists)

    def missing(self) -> list[str]:
        return sorted(cid for cid, ref in self._paths.items() if not ref.exists)

    def summary(self) -> dict:
        per_fold: dict[int, dict] = {}
        for ref in self._paths.values():
            entry = per_fold.setdefault(ref.fold, {"expected": 0, "present": 0})
            entry["expected"] += 1
            entry["present"] += int(ref.exists)
        expected = set(self._paths)
        return {
            "pred_dir": str(self.pred_dir),
            "n_expected": len(self._paths),
            "n_present": len(self.available()),
            "missing_cases": self.missing(),
            "per_fold": per_fold,
            "ambiguous_folds": sorted(self._unknown_folds),
            "splits_sha256": self.splits.sha256 if self.splits else None,
        }

    # ------------------------------------------------------------------- load
    def load(self, case_id: str):
        """Load a NIfTI label volume (requires nibabel; imported lazily)."""
        import nibabel as nib

        ref = self.ref(case_id)
        if not ref.exists:
            raise FileNotFoundError(ref.path)
        return nib.load(str(ref.path)).get_fdata().astype("uint8")

    def __repr__(self) -> str:
        return (f"CVPredictionIndex({self.pred_dir!s}, "
                f"present={len(self.available())}/{len(self)})")
