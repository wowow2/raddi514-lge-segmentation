"""Experiment discovery, explicit output views and saved-result comparison.

experiment.json and output.json are authoritative. Discovery never scans
historical prediction directories or standalone metrics files.
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

from src.config import REPO_ROOT
from src.experiment_registry import ArtifactRoots, catalog_views
from src.layout import canonical_catalog, write_output
from src.datautils.predictions import CVPredictionIndex
from src.datautils.splits import CVSplits
from src.results import ResultsTable

SCHEMA_VERSION = 1

SECTIONS = ("model", "data", "training", "inference", "postprocess",
            "merge", "paths", "status", "provenance", "eval")


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


def read_manifest(path: str | os.PathLike) -> dict[str, Any]:
    """Load a manifest, returning ``{}`` when the file does not exist."""
    path = Path(path)
    if path.resolve().is_relative_to(REPO_ROOT):
        catalog = canonical_catalog(REPO_ROOT)
        if catalog is not None:
            training = path.parent.parent / "training.json"
            if path.parent.name == "checkpoints" and training.exists():
                return json.loads(training.read_text())
            current = json.loads(path.read_text()) if path.is_file() else {}
            for name in (current.get("run_id"), path.parent.name, path.parent.parent.name):
                if name in catalog["outputs"]:
                    return catalog["outputs"][name]
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def write_manifest(path: str | os.PathLike,
                   manifest: "RunManifest | dict[str, Any]") -> Path:
    """Write a manifest with consistent formatting."""
    path = Path(path)
    data = manifest.to_dict() if isinstance(manifest, RunManifest) else manifest
    if path.resolve().is_relative_to(REPO_ROOT):
        destination = write_output(REPO_ROOT, data)
        if destination:
            return destination
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


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
        return self.store.artifact_roots.resolve(value)

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
        path = self.manifest.get("data.splits_path") or self.store.splits_path
        if path is None:
            return None
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
        names = {
            "best": ("checkpoint_best.pt", "checkpoint_best.pth"),
            "final": ("checkpoint_last.pt", "checkpoint_final.pth"),
        }.get(self.manifest.get("inference.checkpoint"),
              ("checkpoint_best.pt", "checkpoint_best.pth",
               "checkpoint_last.pt", "checkpoint_final.pth"))
        for name in names:
            path = self.checkpoint_dir / f"fold_{fold}" / name
            if path.exists():
                return path
        return None

    # -------------------------------------------------------------- status
    def is_complete(self) -> bool | None:
        """OOF coverage; ``None`` when the split or predictions are unknown."""
        index = self.prediction_index()
        if index is None or index.splits is None:
            return None
        summary = index.summary()
        if summary["n_expected"] == 0:
            return None
        return summary["n_present"] == summary["n_expected"]

    def status_summary(self) -> dict:
        splits = self.splits()
        index = self.prediction_index()
        return {
            "has_metrics": self.metrics_path is not None and self.metrics_path.exists(),
            "prediction": index.summary() if index is not None else None,
            "split_hash_match": (splits is None or
                                 self.manifest.get("data.splits_sha256") in (None, splits.sha256)),
            "checkpoints": {fold: self.checkpoint(fold) is not None
                            for fold in range(len(splits.folds) if splits else 5)},
            "legacy": self.manifest.legacy,
        }

    def summary_row(self) -> dict:
        index = self.prediction_index()
        if index is None:
            cases = "?"
        else:
            summary = index.summary()
            cases = (f"{summary['n_present']}/{summary['n_expected']}"
                     if index.splits is not None else f"{summary['n_present']}/?")
        splits = self.splits()
        n_folds = len(splits.folds) if splits is not None else 5
        return {
            "run_id": self.run_id,
            "stage": self.manifest.stage or "?",
            "model": self.manifest.get("model.name") or "?",
            "tags": ",".join(self.manifest.tags),
            "split": (self.manifest.get("data.splits_sha256") or "?")[:8],
            "checkpoints": f"{sum(self.checkpoint(f) is not None for f in range(n_folds))}/{n_folds}",
            "cases": cases,
            "metrics": "yes" if self.metrics_path is not None and self.metrics_path.exists() else "no",
            "legacy": "yes" if self.manifest.legacy else "no",
        }

    def __repr__(self) -> str:
        return f"Run({self.run_id!r}, stage={self.manifest.stage!r})"


class RunStore:
    """Discover canonical experiments and compare saved results."""

    def __init__(self, repo_root: str | os.PathLike | None = None,
                 splits_path: str | os.PathLike | None = None):
        self.repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
        self.artifact_roots = ArtifactRoots(self.repo_root)
        self.splits_path = str(splits_path) if splits_path else None
        self._views: dict[str, Run] = {}
        self._runs: dict[str, Run] | None = None

    def discover(self, refresh: bool = False) -> list[Run]:
        if self._runs is None or refresh:
            catalog = canonical_catalog(self.repo_root)
            if catalog is None:
                self._runs, self._views = {}, {}
            else:
                canonical, views = catalog_views(catalog)
                self._views = {
                    key: Run(RunManifest.from_dict(value), self, sources=[
                        self.repo_root / "experiments" / catalog["membership"][key] / "outputs" / key / "output.json"])
                    for key, value in views.items()}
                self._runs = {
                    key: Run(RunManifest.from_dict(value), self, sources=[
                        self.repo_root / "experiments" / key / "experiment.json"])
                    for key, value in canonical.items()}
        return sorted(self._runs.values(), key=lambda run: run.run_id)

    # -------------------------------------------------------------- queries
    def get(self, run_id: str) -> Run:
        runs = {run.run_id: run for run in self.discover()}
        if run_id in self._views:
            return self._views[run_id]
        if run_id not in runs:
            raise KeyError(
                f"Unknown run {run_id!r}. Known runs: {', '.join(sorted(runs)) or '(none)'}")
        return runs[run_id]

    def select(self, run_ids: Iterable[str] | None = None, **filters: Any) -> list[Run]:
        runs = self.discover()
        if run_ids:
            runs = [self.get(run_id) for run_id in run_ids]
        for key, expected in filters.items():
            if expected is None:
                continue
            runs = [r for r in runs if _matches(r, key, expected)]
        return runs

def _matches(run: Run, key: str, expected: Any) -> bool:
    if key == "model":
        key = "model.name"
    if key == "tag":
        return expected in run.manifest.tags
    if key == "complete":
        return run.is_complete() is bool(expected)
    if key == "has_metrics":
        return (run.metrics_path is not None and run.metrics_path.exists()) is bool(expected)
    if key == "legacy":
        return run.manifest.legacy is bool(expected)
    value = run.manifest.get(key)
    if isinstance(value, str) and isinstance(expected, str):
        return expected.lower() in value.lower()
    return value == expected


def load_runs(**kwargs: Any) -> RunStore:
    """Small convenience wrapper: ``load_runs()`` -> :class:`RunStore`."""
    return RunStore(**kwargs)
