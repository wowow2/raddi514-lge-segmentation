"""Configured artifact roots and canonical experiment/output grouping."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path



class ArtifactRoots:
    def __init__(self, repo_root, config=None):
        self.repo = Path(repo_root).resolve()
        path = self.repo / "config/roots.json"
        self.config = config if config is not None else (
            json.loads(path.read_text()) if path.exists() else {})
        self.profile = os.environ.get("RADDI_PROFILE", self.config.get("profile", "local"))
        self.roots = {"data": "data", "nnunet": "data/nnunet", "experiments": "experiments"}
        self.roots.update(self.config.get("profiles", {}).get(self.profile, {}))

    def resolve(self, value):
        if value is None:
            return None
        value = str(value)
        if "://" in value:
            name, relative = value.split("://", 1)
            base = self.roots.get(name)
            if base is None:
                raise ValueError(f"No {name!r} root for profile {self.profile!r}")
            base = Path(base).expanduser()
            base = base if base.is_absolute() else self.repo / base
            path = (base / relative).resolve()
            if not path.is_relative_to(base.resolve()):
                raise ValueError(f"Artifact escapes configured root: {value}")
            return path
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.repo / path

    def portable(self, value):
        if not isinstance(value, str):
            return value
        if "://" in value:
            return value
        # Translate known historical machine prefixes, never arbitrary basenames.
        for prefix, replacement in sorted(self.config.get("aliases", {}).items(),
                                          key=lambda pair: -len(pair[0])):
            if value.startswith(prefix):
                return replacement + value[len(prefix):]
        path = Path(value)
        if path.is_absolute():
            for name in ("experiments", "nnunet", "data"):
                base = self.roots.get(name)
                if base:
                    base = Path(base).expanduser()
                    base = base if base.is_absolute() else self.repo / base
                    if path.is_relative_to(base):
                        return name + "://" + str(path.relative_to(base))
            return value
        return value


def experiment_id(run_id):
    if run_id in {"503_3d", "503_3d_raw", "503_3d_pp", "503_best",
                  "nnUNet_validation_503_best", "503_best_pp"}:
        return "cascade_503_resenc"
    if run_id.startswith("nnUNet_validation"):
        for suffix in ("_pp_grow1", "_pp", "_grow1"):
            if run_id.endswith(suffix):
                return run_id[:-len(suffix)]
    # Polar outputs share weights; radius inputs require a separate model.
    if run_id.startswith("polar_preds"):
        return "polar_radius" if "radius" in run_id else "polar"
    return run_id


def catalog_views(catalog):
    """Group authoritative output records by experiment and its explicit default."""
    views = copy.deepcopy(catalog["outputs"])
    canonical = {}
    for exp, preferred in catalog["defaults"].items():
        record = copy.deepcopy(views[preferred])
        record.update(run_id=exp, default_output=preferred,
                      outputs=sorted(key for key, owner in catalog["membership"].items() if owner == exp))
        canonical[exp] = record
    return canonical, views
