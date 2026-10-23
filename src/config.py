"""Central application configuration, labels, and path resolution.

Paths default to the repository root, not the current working directory, so
scripts behave the same whether launched from the repo, ``scripts/``, or a
cluster checkout.
"""

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
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _env_path(name: str, default_rel: str | os.PathLike[str]) -> str:
    value = os.getenv(name)
    return str(repo_path(value)) if value else str(repo_path(default_rel))


DEFAULT_GT_ROOT: str = _env_path("EMIDEC_GT_ROOT", "var/emidec_dataset/train")
DEFAULT_PRED_DIR: str = _env_path("NNUNET_PRED_DIR", "var/nnUNet_validation")
DEFAULT_SPLITS_PATH: str = _env_path(
    "NNUNET_SPLITS_PATH", Path(DEFAULT_PRED_DIR) / "splits_final.json"
)
DEFAULT_OUT_PATH: str = _env_path("NNUNET_CV_OUT", "var/results/nnunet_2d_cv_metrics.json")
DEFAULT_STAGE1_PRED_DIR: str = str(repo_path("var/nnUNet_validation_501_pp"))
DEFAULT_POLAR_MODEL_DIR: str = str(repo_path("var/polar_models"))


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
