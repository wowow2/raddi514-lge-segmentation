"""Run manifests, discovery, and a small experiment store.

A run is described by a ``run.json`` manifest written next to its outputs
(prediction directory, checkpoint directory, or results file). Manifests for
the same ``run_id`` are merged, non-null fields winning, so training,
prediction, and evaluation entry points can each contribute what they know
without duplicating a central registry.

Historical directories without a manifest are discovered read-only and marked
``provenance.legacy`` with unknown fields left as ``None``.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.config import DEFAULT_SPLITS_PATH, REPO_ROOT
from src.datautils.predictions import CVPredictionIndex
from src.datautils.splits import CVSplits
from src.results import Comparison, ResultsTable, compare, is_evaluation_report

SCHEMA_VERSION = 1

SECTIONS = ("model", "data", "training", "inference", "postprocess",
            "merge", "paths", "status", "provenance", "eval")

DEFAULT_MANIFEST_ROOTS = ("var/runs", "var/models", "var/results")
DEFAULT_LEGACY_ROOTS = (
    "var/nnUNet_validation*",
    "var/polar_preds*",
    "var/polar_models*",
)

IDENTITY_PATHS = (
    "stage", "model.name", "model.params",
    "data.splits_sha256", "data.label_scheme", "data.channels",
)

ESSENTIAL_UNKNOWN = (
    "model.name", "data.splits_sha256", "training.seed",
    "paths.predictions", "paths.metrics",
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=3, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance_now(command: str | None = None) -> dict:
    """Code/package provenance; unavailable values are ``None`` (unknown)."""
    packages = {}
    for dist in ("torch", "nnunetv2", "nibabel", "numpy", "scipy",
                 "surface-distance", "scikit-learn", "matplotlib"):
        try:
            packages[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            packages[dist] = None
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "created_at": _now(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "command": command,
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
        "packages": packages,
    }


def _empty_list(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _merge_value(primary: Any, secondary: Any) -> Any:
    if isinstance(primary, dict) and isinstance(secondary, dict):
        return _merge_dicts(primary, secondary)
    if isinstance(primary, list) and isinstance(secondary, list):
        return primary or secondary
    return primary if primary is not None else secondary


def _merge_dicts(primary: dict, secondary: dict) -> dict:
    merged = dict(secondary)
    for key, value in primary.items():
        merged[key] = _merge_value(value, secondary.get(key)) if key in secondary else value
    return merged


@dataclass
class RunManifest:
    """Structured metadata for one experiment."""

    run_id: str
    schema_version: int = SCHEMA_VERSION
    description: str | None = None
    display_name: str | None = None
    tags: list[str] = field(default_factory=list)
    stage: str | None = None
    parent_run: str | None = None
    source_run: str | None = None
    model: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    postprocess: dict[str, Any] = field(default_factory=dict)
    merge: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for section in SECTIONS:
            if getattr(self, section) is None:
                setattr(self, section, {})
        self.tags = list(self.tags or [])

    # --------------------------------------------------------------- mapping
    def to_dict(self) -> dict:
        data = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "description": self.description,
            "display_name": self.display_name,
            "tags": sorted(set(self.tags)),
            "stage": self.stage,
            "parent_run": self.parent_run,
            "source_run": self.source_run,
            **{section: getattr(self, section) for section in SECTIONS},
            **self.extra,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RunManifest":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in known and k != "extra"}
        extra = {k: v for k, v in data.items() if k not in known}
        kwargs.setdefault("run_id", data.get("run_id"))
        kwargs.setdefault("tags", [])
        return cls(**kwargs, extra=extra)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up ``section.key.subkey`` through nested dicts/attributes."""
        value: Any = self
        for part in dotted.split("."):
            if value is None:
                return default
            if isinstance(value, dict):
                if part not in value:
                    return default
                value = value[part]
            else:
                if not hasattr(value, part):
                    return default
                value = getattr(value, part)
        return default if value is None else value

    @property
    def legacy(self) -> bool:
        return bool(self.provenance.get("legacy"))

    def identity_conflicts(self, other: "RunManifest") -> list[dict]:
        """Identity fields that disagree between two manifests for one run."""
        conflicts = []
        for path in IDENTITY_PATHS:
            left, right = self.get(path), other.get(path)
            if left is None or right is None:
                continue
            if left != right:
                conflicts.append({"field": path, "kept": left, "other": right})
        return conflicts

    def missing_fields(self) -> list[str]:
        return [key for key in ESSENTIAL_UNKNOWN if _empty_list(self.get(key))]

    def merged_with(self, other: "RunManifest") -> "RunManifest":
        """Return a new manifest where ``self`` wins over ``other``."""
        merged = RunManifest(run_id=self.run_id)
        for name in ("schema_version",):
            setattr(merged, name, max(self.schema_version, other.schema_version))
        for name in ("description", "display_name", "stage", "parent_run", "source_run"):
            setattr(merged, name, _merge_value(getattr(self, name), getattr(other, name)))
        merged.tags = sorted(set(self.tags) | set(other.tags))
        for section in SECTIONS:
            setattr(merged, section, _merge_dicts(
                getattr(self, section), getattr(other, section)))
        merged.extra = _merge_dicts(self.extra, other.extra)
        return merged


class Run:
    """A discovered experiment: metadata plus lazily resolved outputs."""

    def __init__(self, manifest: RunManifest, store: "RunStore",
                 sources: Iterable[Path] = ()):
        self.manifest = manifest
        self.store = store
        self.sources = [Path(p) for p in sources]

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def name(self) -> str:
        return self.manifest.display_name or self.run_id

    def __getattr__(self, item: str) -> Any:
        if item in ("stage", "tags", "parent_run", "source_run", "description"):
            return getattr(self.manifest, item)
        raise AttributeError(item)

    def resolve(self, value: str | os.PathLike | None) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.store.repo_root / path

    @property
    def pred_dir(self) -> Path | None:
        return self.resolve(self.manifest.get("paths.predictions"))

    @property
    def prob_dir(self) -> Path | None:
        return self.resolve(self.manifest.get("paths.probabilities"))

    @property
    def metrics_path(self) -> Path | None:
        return self.resolve(self.manifest.get("paths.metrics"))

    @property
    def checkpoint_dir(self) -> Path | None:
        return self.resolve(self.manifest.get("paths.checkpoints"))

    def splits(self) -> CVSplits | None:
        path = self.manifest.get("data.splits_path")
        if path is None and self.manifest.legacy:
            # Legacy directories may be partial: use a split stored beside the
            # predictions, but never assume the repository default.
            local = self.pred_dir / "splits_final.json" if self.pred_dir else None
            if local is None or not local.exists():
                return None
            path = local
        path = path or self.store.splits_path or DEFAULT_SPLITS_PATH
        resolved = self.resolve(path) or path
        try:
            return CVSplits.load(resolved, validate=False)
        except (OSError, ValueError):
            return None

    def prediction_index(self) -> CVPredictionIndex | None:
        if self.pred_dir is None or not self.pred_dir.exists():
            return None
        splits = self.splits()
        return CVPredictionIndex(
            self.pred_dir, splits=splits, use_default_splits=False)

    def probability_index(self) -> CVPredictionIndex | None:
        if self.prob_dir is None or not self.prob_dir.exists():
            return None
        return CVPredictionIndex(
            self.prob_dir, splits=self.splits(), suffix=".npz",
            use_default_splits=False)

    def load_metrics(self) -> ResultsTable | None:
        if self.metrics_path is None or not self.metrics_path.exists():
            return None
        return ResultsTable.load(self.metrics_path, run_id=self.run_id)

    def load_pred(self, case_id: str):
        index = self.prediction_index()
        if index is None:
            raise FileNotFoundError(f"{self.run_id} has no prediction directory")
        return index.load(case_id)

    def checkpoint(self, fold: int = 0) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        for name in ("checkpoint_best.pt", "checkpoint_best.pth",
                     "checkpoint_last.pt", "checkpoint_final.pth"):
            path = self.checkpoint_dir / f"fold_{fold}" / name
            if path.exists():
                return path
        return None

    # -------------------------------------------------------------- status
    def status(self) -> dict:
        info: dict[str, Any] = {
            "run_id": self.run_id,
            "legacy": self.manifest.legacy,
            "manifest_conflicts": list(
                self.manifest.provenance.get("merge_conflicts") or []),
            "has_metrics": self.metrics_path is not None and self.metrics_path.exists(),
            "checkpoints": {},
            "prediction": None,
            "split_errors": [],
            "split_warnings": [],
        }
        splits = self.splits()
        if splits is not None:
            validation = splits.validate()
            info["split_errors"] = validation.errors
            info["split_warnings"] = validation.warnings
            recorded = self.manifest.get("data.splits_sha256")
            if recorded and recorded != splits.sha256:
                info["split_errors"].append(
                    "recorded splits hash does not match the splits file "
                    f"({recorded[:8]} vs {splits.sha256[:8]})")

        index = self.prediction_index()
        if index is not None:
            summary = index.summary()
            info["prediction"] = summary
            info["n_present"] = summary["n_present"]
            if index.splits is None:
                # Legacy directory without a split: expected coverage unknown.
                info["n_expected"] = None
                info["missing_cases"] = []
                info["complete"] = None
            else:
                info["n_expected"] = summary["n_expected"]
                info["missing_cases"] = summary["missing_cases"]
                info["complete"] = (None if summary["n_expected"] == 0
                                    else summary["n_present"] == summary["n_expected"])
        else:
            info["n_expected"] = None
            info["n_present"] = None
            info["complete"] = None

        prob_index = self.probability_index()
        info["probabilities"] = (prob_index.summary()
                                 if prob_index is not None else None)

        n_folds = len(splits.folds) if splits is not None else 5
        for fold in range(n_folds):
            info["checkpoints"][fold] = self.checkpoint(fold) is not None

        info["fully_tracked"] = not self.manifest.missing_fields()
        return info

    def summary_row(self) -> dict:
        status = self.status()
        return {
            "run_id": self.run_id,
            "stage": self.manifest.stage or "?",
            "model": self.manifest.get("model.name") or "?",
            "tags": ",".join(self.manifest.tags),
            "split": (self.manifest.get("data.splits_sha256") or "?")[:8],
            "checkpoints": f"{sum(status['checkpoints'].values())}/{len(status['checkpoints'])}",
            "cases": (f"{status['n_present']}/{status['n_expected']}"
                      if status["n_expected"] is not None
                      else (f"{status['n_present']}/?" if status["n_present"] is not None
                            else "?")),
            "metrics": "yes" if status["has_metrics"] else "no",
            "legacy": "yes" if status["legacy"] else "no",
            "conflict": "yes" if status["manifest_conflicts"] else "no",
        }

    def __repr__(self) -> str:
        return f"Run({self.run_id!r}, stage={self.manifest.stage!r})"


class RunStore:
    """Discover manifests, legacy outputs, and compare saved results."""

    def __init__(self,
                 roots: Iterable[str | os.PathLike] | None = None,
                 legacy_roots: Iterable[str | os.PathLike] | None = None,
                 repo_root: str | os.PathLike | None = None,
                 splits_path: str | os.PathLike | None = None):
        self.repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
        self.splits_path = str(splits_path) if splits_path else None
        self.roots = self._expand(
            DEFAULT_MANIFEST_ROOTS if roots is None else roots)
        self.legacy_roots = self._expand(
            legacy_roots if legacy_roots is not None else DEFAULT_LEGACY_ROOTS)
        # Manifests are also written beside checkpoints and predictions.
        for path in self.legacy_roots:
            if path not in self.roots:
                self.roots.append(path)
        self.registry_dir = self.repo_root / "var/runs"
        if self.registry_dir not in self.roots:
            self.roots.append(self.registry_dir)
        self._runs: dict[str, Run] | None = None

    # ------------------------------------------------------------ discovery
    def _expand(self, patterns: Iterable[str | os.PathLike]) -> list[Path]:
        found: list[Path] = []
        for pattern in patterns:
            raw = Path(pattern).expanduser()
            if raw.is_absolute():
                found.append(raw)
                continue
            if any(ch in str(pattern) for ch in "*?["):
                found.extend(sorted(self.repo_root.glob(str(pattern))))
            else:
                found.append(self.repo_root / raw)
        return [p for p in found if p.exists()]

    @staticmethod
    def _manifest_files(root: Path) -> list[Path]:
        if root.is_dir():
            return sorted(root.rglob("run.json"))
        return [root] if root.name == "run.json" else []

    def _collect_manifests(self) -> dict[str, list[tuple[Path, RunManifest]]]:
        collected: dict[str, list[tuple[Path, RunManifest]]] = {}
        for root in self.roots:
            for path in self._manifest_files(root):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    manifest = RunManifest.from_dict(data)
                except (OSError, json.JSONDecodeError, TypeError):
                    continue
                if not manifest.run_id:
                    continue
                collected.setdefault(manifest.run_id, []).append((path, manifest))
        return collected

    @staticmethod
    def _has_folds(path: Path) -> bool:
        """True for prediction trees, not trainer dirs that only hold logs."""
        return path.is_dir() and next(path.glob("fold_*/validation/*.nii.gz"), None) is not None

    def _legacy_prediction_dirs(self) -> list[Path]:
        found: list[Path] = []
        for root in self.legacy_roots:
            candidates = [root]
            candidates += sorted(p for p in root.glob("*") if p.is_dir())
            candidates += sorted(p for p in root.glob("*/*") if p.is_dir())
            for path in candidates:
                if self._has_folds(path) and not (path / "run.json").exists():
                    if path not in found:
                        found.append(path)
        return found

    def _legacy_metric_files(self) -> list[Path]:
        results_dir = self.repo_root / "var/results"
        if not results_dir.is_dir():
            return []
        return sorted(results_dir.glob("*.json"))

    def discover(self, refresh: bool = False) -> list[Run]:
        if self._runs is not None and not refresh:
            return sorted(self._runs.values(), key=lambda r: r.run_id)

        collected = self._collect_manifests()
        runs: dict[str, Run] = {}
        sources: dict[str, list[Path]] = {}

        for run_id, entries in collected.items():
            entries = sorted(entries, key=lambda item: str(item[0]))
            manifest = entries[0][1]
            primary_path = entries[0][0]
            conflicts: list[dict] = []
            for other_path, other in entries[1:]:
                for conflict in manifest.identity_conflicts(other):
                    conflicts.append({
                        **conflict,
                        "sources": [str(primary_path), str(other_path)],
                    })
                manifest = manifest.merged_with(other)
            if conflicts:
                manifest.provenance["merge_conflicts"] = conflicts
            runs[run_id] = Run(manifest, self)
            sources[run_id] = [path for path, _ in entries]

        # Read-only fallback for prediction directories without run.json.
        # Skip a directory already registered/claimed by another run.
        claimed_predictions = {
            run.pred_dir.resolve() for run in runs.values() if run.pred_dir is not None}
        for path in self._legacy_prediction_dirs():
            if path.resolve() in claimed_predictions:
                continue
            run_id = self._unique_id(path.name, runs)
            manifest = RunManifest(
                run_id=run_id,
                provenance={"legacy": True, "created_at": _now(),
                            "source": "prediction directory (no run.json)"},
                paths={"predictions": self._rel(path)},
            )
            runs[run_id] = Run(manifest, self)
            sources[run_id] = [path]

        # Read-only fallback for historical metrics files.
        for path in self._legacy_metric_files():
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not is_evaluation_report(data):
                continue
            claimed_metrics = {
                run.metrics_path.resolve() for run in runs.values()
                if run.metrics_path is not None}
            if path.resolve() in claimed_metrics:
                continue
            existing = runs.get(path.stem)
            if existing is not None:
                if existing.manifest.get("paths.metrics") is None:
                    existing.manifest.paths["metrics"] = self._rel(path)
                    sources.setdefault(path.stem, []).append(path)
                continue
            manifest = RunManifest(
                run_id=path.stem,
                provenance={"legacy": True, "created_at": _now(),
                            "source": "metrics file (no run.json)"},
                paths={"metrics": self._rel(path)},
            )
            runs[path.stem] = Run(manifest, self)
            sources[path.stem] = [path]

        for run_id, run in runs.items():
            run.sources = sources.get(run_id, [])
        self._runs = runs
        return sorted(runs.values(), key=lambda r: r.run_id)

    @staticmethod
    def _unique_id(base: str, existing: dict) -> str:
        if base not in existing:
            return base
        suffix = 2
        while f"{base}_{suffix}" in existing:
            suffix += 1
        return f"{base}_{suffix}"

    def _rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.repo_root.resolve()))
        except ValueError:
            return str(path)

    # -------------------------------------------------------------- queries
    def get(self, run_id: str) -> Run:
        runs = {run.run_id: run for run in self.discover()}
        if run_id not in runs:
            raise KeyError(
                f"Unknown run {run_id!r}. Known runs: {', '.join(sorted(runs)) or '(none)'}")
        return runs[run_id]

    def select(self, run_ids: Iterable[str] | None = None, **filters: Any) -> list[Run]:
        runs = self.discover()
        if run_ids:
            wanted = list(run_ids)
            missing = sorted(set(wanted) - {r.run_id for r in runs})
            if missing:
                raise KeyError(f"Unknown runs: {', '.join(missing)}")
            order = {run_id: i for i, run_id in enumerate(wanted)}
            runs = sorted((r for r in runs if r.run_id in order),
                          key=lambda r: order[r.run_id])
        for key, expected in filters.items():
            if expected is None:
                continue
            runs = [r for r in runs if _matches(r, key, expected)]
        return runs

    # ------------------------------------------------------------- register
    def register(self, run_id: str, **fields: Any) -> Run:
        """Write/merge an explicit manifest under ``var/runs/<run_id>``."""
        path = self.registry_dir / run_id / "run.json"
        existing = None
        if path.exists():
            with open(path) as f:
                existing = RunManifest.from_dict(json.load(f))
        manifest = RunManifest.from_dict({"run_id": run_id, **fields})
        if existing is not None:
            conflicts = manifest.identity_conflicts(existing)
            if conflicts:
                detail = "; ".join(
                    f"{c['field']}: {c['kept']!r} vs {c['other']!r}"
                    for c in conflicts)
                raise ValueError(
                    f"Refusing to register {run_id}: identity conflicts with "
                    f"{path} ({detail})")
            manifest = manifest.merged_with(existing)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        self._runs = None
        return Run(manifest, self, sources=[path])

    # -------------------------------------------------------------- compare
    def compare(self, run_ids: Iterable[str] | None = None, metric: str = "dice",
                case_set: str = "intersection",
                require_same_cases: bool = False,
                check_splits: bool = True) -> Comparison:
        runs = self.select(run_ids)
        conflicted = [r.run_id for r in runs
                      if r.manifest.provenance.get("merge_conflicts")]
        if conflicted:
            raise ValueError(
                "Conflicting manifests for: " + ", ".join(conflicted)
                + " (see `scripts/runs.py show <run>` for source files)")
        if check_splits:
            hashes = {r.run_id: r.manifest.get("data.splits_sha256")
                      for r in runs}
            known = {h for h in hashes.values() if h}
            if len(known) > 1:
                detail = ", ".join(f"{k}={v[:8]}" for k, v in hashes.items() if v)
                raise ValueError(f"Runs used different splits: {detail}")

        tables: dict[str, ResultsTable] = {}
        for run in runs:
            table = run.load_metrics()
            if table is None:
                raise FileNotFoundError(f"{run.run_id} has no metrics file")
            tables[run.run_id] = table
        return compare(tables, metric=metric, case_set=case_set,
                       require_same_cases=require_same_cases)


def _matches(run: Run, key: str, expected: Any) -> bool:
    if key == "model":
        key = "model.name"
    if key == "tag":
        return expected in run.manifest.tags
    if key == "complete":
        return run.status()["complete"] is bool(expected)
    if key == "has_metrics":
        return run.status()["has_metrics"] is bool(expected)
    if key == "legacy":
        return run.manifest.legacy is bool(expected)
    value = run.manifest.get(key)
    if isinstance(value, str) and isinstance(expected, str):
        return expected.lower() in value.lower()
    return value == expected


def load_runs(**kwargs: Any) -> RunStore:
    """Small convenience wrapper: ``load_runs()`` -> :class:`RunStore`."""
    return RunStore(**kwargs)
