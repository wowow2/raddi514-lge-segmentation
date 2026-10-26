"""Central application configuration, labels, and path resolution.

Paths default to the repository root, not the current working directory, so
scripts behave the same whether launched from the repo, ``scripts/``, or a
cluster checkout.
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

env_file = REPO_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

LABEL_BACKGROUND = 0
LABEL_CAVITY = 1
LABEL_MYOCARDIUM = 2
LABEL_SCAR = 3
LABEL_MVO = 4

EVAL_LABELS: tuple[int, ...] = (LABEL_CAVITY, LABEL_MYOCARDIUM, LABEL_SCAR, LABEL_MVO)

LABEL_NAMES: dict[int, str] = {
    LABEL_BACKGROUND: "Background",
    LABEL_CAVITY: "Cavity",
    LABEL_MYOCARDIUM: "Myo",
    LABEL_SCAR: "Scar",
    LABEL_MVO: "MVO",
}


def repo_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a path relative to the repository root (absolute paths pass through)."""
    from src.experiment_registry import ArtifactRoots
    return ArtifactRoots(REPO_ROOT).resolve(value)


def artifact_default(output_id: str, field: str, fallback: str) -> str:
    from src.layout import default_path
    return str(default_path(REPO_ROOT, output_id, field, fallback))


def _env_path(name: str, default_rel: str | os.PathLike[str]) -> str:
    value = os.getenv(name)
    return str(repo_path(value)) if value else str(repo_path(default_rel))


DEFAULT_GT_ROOT: str = _env_path("EMIDEC_GT_ROOT", "data://source/emidec/train")
DEFAULT_PRED_DIR: str = _env_path("NNUNET_PRED_DIR", artifact_default(
    "nnUNet_validation", "predictions",
    "experiments://nnUNet_validation/outputs/nnUNet_validation/predictions"))
_layout = REPO_ROOT / "experiments/layout.json"
_split_default = (json.loads(_layout.read_text())["default_splits"] if _layout.exists() else
    "data://splits/emidec-1c67734e21eda69fd2a1e4de053c3923e72997d554f5d071b0a7ce1a16c7bb8b.json")
DEFAULT_SPLITS_PATH: str = _env_path("NNUNET_SPLITS_PATH", _split_default)
DEFAULT_OUT_PATH: str = _env_path("NNUNET_CV_OUT", artifact_default(
    "nnUNet_validation", "metrics",
    "experiments://nnUNet_validation/outputs/nnUNet_validation/metrics.json"))
DEFAULT_STAGE1_PRED_DIR: str = artifact_default("nnUNet_validation_501_pp", "predictions",
    "experiments://nnUNet_validation_501/outputs/nnUNet_validation_501_pp/predictions")
DEFAULT_POLAR_MODEL_DIR: str = str(repo_path("experiments://polar/checkpoints"))
DEFAULT_TEST_ROOT: str = str(repo_path("data://source/emidec/test"))
DEFAULT_NNUNET_RAW: str = str(repo_path("nnunet://nnUNet_raw"))
DEFAULT_REPORT_DIR: str = str(REPO_ROOT / "reports")
DEFAULT_CACHE_DIR: str = str(repo_path("data://cache"))


def nnunet_id_to_case_id(nnunet_id: str) -> str:
    """``EMIDEC_P001`` -> ``Case_P001`` (idempotent for ``Case_*``)."""
    if nnunet_id.startswith("Case_"):
        return nnunet_id
    return "Case_" + nnunet_id.removeprefix("EMIDEC_")


def case_id_to_nnunet_id(case_id: str) -> str:
    """``Case_P001`` -> ``EMIDEC_P001`` (idempotent for ``EMIDEC_*``)."""
    if case_id.startswith("EMIDEC_"):
        return case_id
    return "EMIDEC_" + case_id.removeprefix("Case_")
