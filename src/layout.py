"""Canonical experiment metadata, immutable splits and artifact verification."""
from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path

from src.experiment_registry import ArtifactRoots, experiment_id

MARKER = "experiments/layout.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid experiment/output ID: {value!r}")
    return value


def canonical_catalog(repo):
    repo = Path(repo)
    outputs, membership, defaults = {}, {}, {}
    if not (repo / "experiments").is_dir():
        return None
    for path in sorted((repo / "experiments").glob("*/experiment.json")):
        experiment = read_json(path)
        exp = safe_id(experiment["experiment_id"])
        if exp != path.parent.name:
            raise ValueError(f"Experiment ID does not match directory: {path}")
        if len(experiment["outputs"]) != len(set(experiment["outputs"])):
            raise ValueError(f"Duplicate output IDs: {exp}")
        for name in experiment["outputs"]:
            name = safe_id(name)
            if name in outputs:
                raise ValueError(f"Output belongs to more than one experiment: {name}")
            output = read_json(path.parent / "outputs" / name / "output.json")
            if output.get("run_id") != name:
                raise ValueError(f"Output ID mismatch in {path.parent / 'outputs' / name}")
            outputs[name] = output
            membership[name] = exp
        if not experiment["outputs"]:
            continue
        default = experiment["default_output"]
        if default not in experiment["outputs"]:
            raise ValueError(f"Unknown default output: {exp}/{default}")
        defaults[exp] = default
    return {"outputs": outputs, "membership": membership, "defaults": defaults}


def default_path(repo, output_id, field, fallback):
    catalog = canonical_catalog(repo)
    value = (catalog or {}).get("outputs", {}).get(output_id, {}).get("paths", {}).get(field)
    return ArtifactRoots(repo).resolve(value or fallback)


def artifact_id(value, repo=None):
    """Output identity must not become the generic directory name 'predictions'."""
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    path = Path(value).resolve()
    try:
        parts = path.relative_to(repo / "experiments").parts
    except ValueError:
        return path.name
    if len(parts) >= 3 and parts[1] == "outputs":
        return parts[2]
    if len(parts) >= 2 and parts[1] == "checkpoints":
        return parts[0] + "__training"
    return path.name


def write_output(repo, manifest):
    """Write canonical metadata when experiment storage exists, otherwise return False."""
    from src.datautils.splits import CVSplits

    repo = Path(repo)
    catalog = canonical_catalog(repo)
    if catalog is None:
        return False
    output = copy.deepcopy(manifest)
    name = safe_id(output["run_id"])
    exp = catalog["membership"].get(name)
    checkpoints = output.get("paths", {}).get("checkpoints")
    if exp is None and checkpoints:
        roots = ArtifactRoots(repo)
        try:
            parts = roots.resolve(checkpoints).resolve().relative_to(repo / "experiments").parts
            if len(parts) == 2 and parts[1] == "checkpoints":
                exp = parts[0]
        except ValueError:
            pass
        owners = {catalog["membership"][key] for key, record in catalog["outputs"].items()
                  if record.get("paths", {}).get("checkpoints") and
                  roots.resolve(record["paths"]["checkpoints"]).resolve() == roots.resolve(checkpoints).resolve()}
        if len(owners) == 1:
            exp = owners.pop()
    if exp is None:
        parent = output.get("parent_run")
        exp = catalog["membership"].get(parent, experiment_id(name))
    safe_id(exp)
    roots = ArtifactRoots(repo)
    for key, value in output.setdefault("paths", {}).items():
        if value:
            output["paths"][key] = roots.portable(str(roots.resolve(value).resolve()))
    data = output.setdefault("data", {})
    split_path = data.get("splits_path")
    if split_path:
        source = roots.resolve(split_path)
        digest = CVSplits.load(source).sha256
        if data.get("splits_sha256") not in (None, digest):
            raise ValueError(f"Split hash mismatch for {name}")
        canonical = repo / "data/splits" / f"emidec-{digest}.json"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        if not canonical.exists():
            shutil.copy2(source, canonical)
        if CVSplits.load(canonical).sha256 != digest:
            raise ValueError(f"Canonical split changed: {canonical}")
        data.update(splits_path=f"data://splits/{canonical.name}", splits_sha256=digest)
    for key in ("gt_root", "stage1_predictions"):
        if data.get(key):
            data[key] = roots.portable(str(roots.resolve(data[key]).resolve()))
    path = repo / "experiments" / exp / "experiment.json"
    experiment = read_json(path) if path.exists() else {
        "schema_version": 1, "experiment_id": exp, "stage": output.get("stage"),
        "model": output.get("model", {}), "outputs": [], "default_output": name}
    if checkpoints and not output["paths"].get("predictions") and not output["paths"].get("metrics"):
        if not experiment["outputs"]:
            experiment["default_output"] = None
        write_json(path.parent / "training.json", output)
        experiment.update(model=output.get("model", {}), training=output.get("training", {}))
        write_json(path, experiment)
        return path.parent / "training.json"
    if name not in experiment["outputs"]:
        experiment["outputs"].append(name)
    write_json(path.parent / "outputs" / name / "output.json", output)
    write_json(path, experiment)
    return path.parent / "outputs" / name / "output.json"


def check_outputs(repo, registry=None, roots=None):
    """Raise on split drift or missing OOF predictions/probabilities/metrics."""
    from src.datautils.splits import CVSplits

    repo = Path(repo)
    registry = registry if registry is not None else canonical_catalog(repo)
    if registry is None:
        raise ValueError("Canonical layout is not active")
    roots = roots or ArtifactRoots(repo)
    for name, output in registry["outputs"].items():
        data = output["data"]
        splits = CVSplits.load(roots.resolve(data["splits_path"]))
        if splits.sha256 != data.get("splits_sha256"):
            raise ValueError(f"Split hash mismatch: {name}")
        for fold, split in enumerate(splits.folds):
            for case in split["val"]:
                case = case.replace("Case_", "EMIDEC_")
                predictions = roots.resolve(output.get("paths", {}).get("predictions"))
                if predictions is not None and not (predictions / f"fold_{fold}/validation/{case}.nii.gz").is_file():
                    raise ValueError(f"Missing OOF prediction: {name}/fold_{fold}/{case}")
                probabilities = roots.resolve(output.get("paths", {}).get("probabilities"))
                if probabilities is not None and not (probabilities / f"fold_{fold}/validation/{case}.npz").is_file():
                    raise ValueError(f"Missing OOF probability: {name}/fold_{fold}/{case}")
        metrics = roots.resolve(output.get("paths", {}).get("metrics"))
        if metrics is not None and not metrics.is_file():
            raise ValueError(f"Missing metrics: {name}: {metrics}")
    return len(registry["defaults"]), len(registry["outputs"])
