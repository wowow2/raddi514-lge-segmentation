"""Export an ROI 3D cascade dataset in nnU-Net raw layout."""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from src.analysis.probs import ProbIndex
from src.config import (
    DEFAULT_GT_ROOT,
    DEFAULT_SPLITS_PATH,
    nnunet_id_to_case_id,
    repo_path,
)
from src.datautils.emidec_dataset import EMIDECDataset
from src.datautils.splits import CVSplits

from .geometry import (
    CropBox,
    crop_array,
    foreground_bbox,
    largest_component,
    shifted_affine,
)

PROB_CHANNELS = ("P_background", "P_cavity", "P_myo", "P_scar", "P_mvo")


@dataclass(frozen=True)
class ExportConfig:
    out_dir: str | Path
    task_name: str
    prob_dir: str | Path
    splits_path: str | Path = DEFAULT_SPLITS_PATH
    gt_root: str | Path = DEFAULT_GT_ROOT
    pad: int = 16
    min_size: int = 64
    multiple: int = 16
    include_probs: bool = True
    overwrite: bool = False
    limit: int | None = None
    case_ids: tuple[str, ...] | None = None


@dataclass
class CaseExport:
    case_id: str
    bbox: dict
    crop_shape: list[int]
    slice_count: int
    gt_coverage: float


@dataclass
class ExportSummary:
    task_name: str
    prob_dir: str
    splits_path: str
    splits_sha256: str
    pad: int
    min_size: int
    multiple: int
    include_probs: bool
    box_source: str
    n_cases: int
    channels: dict[int, str]
    cases: list[CaseExport]
    note: str

    def as_dict(self) -> dict:
        return asdict(self)


def _resolve(value: str | Path) -> Path:
    return Path(repo_path(value))


def _prepare_task_dir(task_dir: Path, overwrite: bool) -> None:
    if task_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{task_dir} exists (pass --overwrite)")
        shutil.rmtree(task_dir)
    (task_dir / "imagesTr").mkdir(parents=True)
    (task_dir / "labelsTr").mkdir(parents=True)


def _normalise_case_ids(case_ids: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if case_ids is None:
        return None
    return tuple(nnunet_id_to_case_id(case_id) for case_id in case_ids)


def _select_cases(dataset: EMIDECDataset, config: ExportConfig) -> list:
    wanted = _normalise_case_ids(config.case_ids)
    cases = dataset.cases
    if wanted is not None:
        wanted_set = set(wanted)
        cases = [case for case in cases if case.case_id in wanted_set]
        missing = sorted(wanted_set - {case.case_id for case in cases})
        if missing:
            raise KeyError(f"unknown case id(s): {', '.join(missing)}")
    if config.limit is not None:
        cases = cases[: config.limit]
    return cases


def _channel_names(include_probs: bool) -> dict[int, str]:
    names = {0: "LGE"}
    if include_probs:
        names.update({i + 1: name for i, name in enumerate(PROB_CHANNELS)})
    return names


def _write_case(case, prob: np.ndarray, box: CropBox, include_probs: bool,
                task_dir: Path) -> tuple[CaseExport, dict]:
    from src.datautils.nnunet_export import NnunetExporter

    if not case.has_gt:
        raise ValueError(f"{case.case_id} has no ground truth")
    image, gt = case.load()
    image_nii = nib.load(case.img_path)
    gt_nii = nib.load(case.gt_path)

    image_crop = crop_array(image, box)
    gt_crop = np.rint(crop_array(gt, box)).astype(np.uint8)
    gt_total = int((gt > 0).sum())
    coverage = float((gt_crop > 0).sum() / gt_total) if gt_total else 1.0
    if coverage < 1.0:
        warnings.warn(
            f"{case.case_id}: crop keeps {coverage:.3f} of GT foreground; "
            "increase --pad if this is not intended",
            RuntimeWarning,
            stacklevel=2,
        )

    channels = [image_crop]
    if include_probs:
        channels.extend(crop_array(prob, box))

    image_affine = shifted_affine(NnunetExporter._spaced_affine(image_nii), box)
    gt_affine = shifted_affine(NnunetExporter._spaced_affine(gt_nii), box)
    nid = NnunetExporter.sanitize_id(case.case_id)
    for index, channel in enumerate(channels):
        nib.save(
            nib.Nifti1Image(channel.astype(np.float32, copy=False), image_affine),
            str(task_dir / "imagesTr" / f"{nid}_{index:04d}.nii.gz"),
        )
    nib.save(
        nib.Nifti1Image(gt_crop, gt_affine),
        str(task_dir / "labelsTr" / f"{nid}.nii.gz"),
    )
    entry = {
        "image": f"./imagesTr/{nid}_0000.nii.gz",
        "label": f"./labelsTr/{nid}.nii.gz",
    }
    case_export = CaseExport(
        case_id=case.case_id,
        bbox=box.as_dict(),
        crop_shape=list(image_crop.shape),
        slice_count=int(image_crop.shape[2]),
        gt_coverage=coverage,
    )
    return case_export, entry


def _write_dataset_json(task_dir: Path, channel_names: dict[int, str],
                        training: list[dict]) -> None:
    from nnunetv2.dataset_conversion.generate_dataset_json import (
        generate_dataset_json,
    )

    from src.datautils.nnunet_export import LABEL_NAMES

    generate_dataset_json(
        str(task_dir),
        channel_names=channel_names,
        labels=dict(LABEL_NAMES),
        num_training_cases=len(training),
        file_ending=".nii.gz",
        training=training,
        license="CC BY-NC-SA 4.0",
        converted_by="scripts/export_cascade.py (EMIDEC ROI cascade)",
        description="EMIDEC LGE short-axis ROI crops with OOF 2D stage-1 "
                    "probability channels",
    )


def export_dataset(config: ExportConfig) -> ExportSummary:
    """Write the dataset and return a summary. Probabilities are always used for
    the ROI; ``include_probs`` only controls whether they are input channels."""
    out_dir = _resolve(config.out_dir) / config.task_name
    prob_dir = _resolve(config.prob_dir)
    splits_path = _resolve(config.splits_path)
    gt_root = _resolve(config.gt_root)
    _prepare_task_dir(out_dir, config.overwrite)

    splits = CVSplits.load(splits_path)
    dataset = EMIDECDataset(str(gt_root))
    prob_index = ProbIndex(str(prob_dir), str(splits_path))
    missing = sorted(set(dataset.case_ids) - set(prob_index.available()))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} case(s) have no OOF probabilities in {prob_dir} "
            f"(first: {missing[0]})")

    cases = []
    training = []
    for case in _select_cases(dataset, config):
        prob = prob_index.load(case.case_id)
        mask = largest_component(prob.argmax(0) > 0)
        box = foreground_bbox(mask, pad=config.pad, min_size=config.min_size,
                              multiple=config.multiple)
        case_export, entry = _write_case(
            case, prob, box, config.include_probs, out_dir)
        cases.append(case_export)
        training.append(entry)

    channel_names = _channel_names(config.include_probs)
    _write_dataset_json(out_dir, channel_names, training)
    shutil.copyfile(splits_path, out_dir / "splits_final.json")

    summary = ExportSummary(
        task_name=config.task_name,
        prob_dir=str(prob_dir),
        splits_path=str(splits_path),
        splits_sha256=splits.sha256,
        pad=config.pad,
        min_size=config.min_size,
        multiple=config.multiple,
        include_probs=config.include_probs,
        box_source="largest 3D component of stage-1 foreground",
        n_cases=len(cases),
        channels=channel_names,
        cases=cases,
        note="Copy splits_final.json into nnUNet_preprocessed/<task>/ after "
             "planning so training uses the stage-1 folds.",
    )
    with open(out_dir / "cascade_manifest.json", "w") as f:
        json.dump(summary.as_dict(), f, indent=2)
    return summary
