"""Central application configuration, labels, and path resolution."""

import os
from pathlib import Path

env_file = Path(".env")
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

DEFAULT_GT_ROOT: str = os.getenv("EMIDEC_GT_ROOT", "emidec_dataset/train")
DEFAULT_PRED_DIR: str = os.getenv("NNUNET_PRED_DIR", "nnUNet_validation")
DEFAULT_SPLITS_PATH: str = os.getenv(
    "NNUNET_SPLITS_PATH", os.path.join(DEFAULT_PRED_DIR, "splits_final.json")
)
DEFAULT_OUT_PATH: str = os.getenv("NNUNET_CV_OUT", "results/nnunet_2d_cv_metrics.json")


def nnunet_id_to_case_id(nnunet_id: str) -> str:
    """EMIDEC_P001 -> Case_P001."""
    return "Case_" + nnunet_id.replace("EMIDEC_", "")


def case_id_to_nnunet_id(case_id: str) -> str:
    """Case_P001 -> EMIDEC_P001."""
    return "EMIDEC_" + case_id.replace("Case_", "")