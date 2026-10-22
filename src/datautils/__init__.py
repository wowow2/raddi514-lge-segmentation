"""Data utilities.

nnU-Net-dependent exports are loaded lazily so importing the dataset loader,
metrics, post-processing, or splits does not require ``nnunetv2`` to be
installed (e.g. a lightweight local analysis environment).
"""

from .emidec_dataset import EMIDECDataset
from .emidec_loader import EMIDECCase, parse_metadata_file
from .metrics import SegmentationMetrics
from .postprocess import (
    MIN_MVO_VOXELS,
    MIN_SCAR_VOXELS,
    MVO_LABEL,
    SCAR_LABEL,
    PostProcessor,
    apply_to_dir,
)
from .splits import CVSplits, StratifiedFolds
from .predictions import CVPredictionIndex

_LAZY = {
    "LABEL_NAMES": "nnunet_export",
    "REGION_CLASS_ORDER": "nnunet_export",
    "REGION_LABEL_NAMES": "nnunet_export",
    "NeighborStacker": "nnunet_export",
    "NnunetExporter": "nnunet_export",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)


__all__ = [
    "EMIDECCase",
    "EMIDECDataset",
    "parse_metadata_file",
    "StratifiedFolds",
    "CVSplits",
    "CVPredictionIndex",
    "SegmentationMetrics",
    "PostProcessor",
    "apply_to_dir",
    "SCAR_LABEL",
    "MIN_SCAR_VOXELS",
    "MVO_LABEL",
    "MIN_MVO_VOXELS",
    "LABEL_NAMES",
    "REGION_LABEL_NAMES",
    "REGION_CLASS_ORDER",
    "NnunetExporter",
    "NeighborStacker",
]
