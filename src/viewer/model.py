"""Data and navigation state for the Qt viewer.

This module deliberately has **no Qt imports**: it can be exercised from the
standard-library test runner and reused by scripts or notebooks. The Qt tabs
are thin renderers over :class:`ViewerModel`.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi

from src.analysis.probs import ProbIndex
from src.config import (DEFAULT_GT_ROOT, DEFAULT_TEST_ROOT, EVAL_LABELS, LABEL_NAMES, LABEL_SCAR)
from src.datautils.emidec_loader import EMIDECCase
from src.runs import RunStore
from src.layout import canonical_catalog

TEST_ROOT = Path(DEFAULT_TEST_ROOT)

SPLIT_ROOTS = {
    "train": [str(DEFAULT_GT_ROOT)],
    "test": [str(TEST_ROOT)],
    "all": [str(DEFAULT_GT_ROOT), str(TEST_ROOT)],
}

# Display-only transforms. Names are what the viewer shows; values are keys.
CHANNELS: "OrderedDict[str, str]" = OrderedDict([
    ("LGE", "lge"),
    ("Window p5-95", "window"),
    ("Local contrast", "dog"),
    ("Edges (LoG)", "log"),
    ("Local std", "std"),
])

CASE_FILTERS = ("All cases", "Pathological", "Normal", "Has scar", "Has MVO")
CASE_SORTS = ("Case ID", "GT scar (high)", "GT scar (low)")

def contour_color(rgb: tuple) -> tuple:
    """Prediction outline color: class hue lifted toward white so it stays
    readable against its own translucent GT fill in Both mode."""
    r, g, b = rgb[:3]
    return tuple(min(1.0, 0.65 * c + 0.35) for c in (r, g, b))


_OVERLAY_COLORS = {
    "fp": (1.00, 0.58, 0.20, 0.75),
    "fn": (0.10, 0.78, 1.00, 0.80),
}


def _norm01(volume: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Percentile-normalize a whole volume to [0, 1]."""
    vmin, vmax = np.percentile(volume, lo), np.percentile(volume, hi)
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - vmin) / (vmax - vmin), 0, 1).astype(np.float32)


def derived_channel(image: np.ndarray, kind: str) -> np.ndarray:
    """Return a display-ready (X, Y, Z) float volume for ``kind``.

    ``image`` is the raw (un-normalized) LGE volume so that filters operate in
    scanner units. Every result is normalized to [0, 1] for imshow.
    """
    image = np.asarray(image, dtype=np.float32)
    if kind == "lge":
        return _norm01(image)
    if kind == "window":
        return _norm01(image, 5, 95)
    if kind == "blur":
        return _norm01(ndi.gaussian_filter(image, sigma=1.2))
    if kind == "dog":
        smooth = ndi.gaussian_filter(image, sigma=3.0)
        return _norm01(image - smooth)
    if kind == "log":
        return _norm01(-ndi.gaussian_laplace(image, sigma=1.0))
    if kind == "std":
        mean = ndi.gaussian_filter(image, sigma=2.0)
        sq = ndi.gaussian_filter(image * image, sigma=2.0)
        return _norm01(np.sqrt(np.maximum(sq - mean * mean, 0.0)))
    raise KeyError(f"unknown channel {kind!r}")


def contour_mask(mask: np.ndarray) -> np.ndarray:
    """Boolean edge of a binary mask (1 voxel, in-plane, per slice)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask)
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    return mask & ~eroded


def label_rgba(label_slice: np.ndarray, visible: Iterable[int],
               colors: dict[int, tuple], alpha: float = 1.0) -> np.ndarray:
    """RGBA overlay for a visible label slice (zero elsewhere)."""
    out = np.zeros((*label_slice.shape, 4), dtype=np.float32)
    for label in visible:
        color = colors.get(int(label))
        if color is None:
            continue
        mask = label_slice == label
        if not mask.any():
            continue
        out[mask] = (color[0], color[1], color[2], float(color[3]) * alpha)
    return out


def error_rgba(gt_slice: np.ndarray, pred_slice: np.ndarray,
               visible: Iterable[int], colors: dict[int, tuple],
               alpha: float = 1.0) -> np.ndarray:
    """TP/FP/FN overlay for the visible classes.

    TP uses each class colour; false positives are orange and false negatives
    cyan. Background disagreements are ignored.
    """
    out = np.zeros((*gt_slice.shape, 4), dtype=np.float32)
    visible = [int(v) for v in visible]
    for label in visible:
        gt = gt_slice == label
        pred = pred_slice == label
        tp = gt & pred
        if tp.any():
            color = colors.get(label)
            if color is not None:
                out[tp] = (color[0], color[1], color[2], float(color[3]) * 0.45 * alpha)
        fp = pred & ~gt
        fn = gt & ~pred
        if fp.any():
            out[fp] = _OVERLAY_COLORS["fp"][:3] + (_OVERLAY_COLORS["fp"][3] * alpha,)
        if fn.any():
            out[fn] = _OVERLAY_COLORS["fn"][:3] + (_OVERLAY_COLORS["fn"][3] * alpha,)
    return out


def probability_rgba(prob_slice: np.ndarray, color: tuple[float, float, float],
                     alpha: float = 0.85, gamma: float = 0.65) -> np.ndarray:
    """Heat-style RGBA overlay from a probability map in [0, 1].

    ``gamma < 1`` lifts low probabilities so weak evidence stays visible.
    """
    p = np.clip(np.asarray(prob_slice, dtype=np.float32), 0, 1)
    out = np.zeros((*p.shape, 4), dtype=np.float32)
    out[..., 0] = color[0]
    out[..., 1] = color[1]
    out[..., 2] = color[2]
    out[..., 3] = np.power(p, gamma) * alpha
    return out


def montage_cols(n: int) -> int:
    """Pick a sensible column count for a whole-volume montage."""
    if n <= 4:
        return 2
    if n <= 6:
        return 3
    if n <= 10:
        return 4
    return 5


def bootstrap_ci(values: Iterable[float], n_boot: int = 1000,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap CI (deterministic seed)."""
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                      dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return mean, float(lo), float(hi)


@dataclass(frozen=True)
class RunChoice:
    run_id: str
    name: str
    stage: str
    has_metrics: bool


class ViewerModel:
    """Dataset + run discovery + caches. All methods are UI-agnostic."""

    def __init__(self, split: str = "all", run_ids: Iterable[str] | None = None,
                 use_predictions: bool = True, cache_cases: int = 8,
                 cache_preds: int = 16):
        if split not in SPLIT_ROOTS:
            raise ValueError(f"unknown split {split!r}")
        self.split = split
        self.use_predictions = use_predictions
        self._cache_cases = int(cache_cases)
        self._cache_preds = int(cache_preds)

        roots = SPLIT_ROOTS[split]
        case_dirs = [d for root in roots if Path(root).is_dir()
                     for d in sorted(Path(root).glob("Case_*")) if d.is_dir()]
        if not case_dirs:
            raise ValueError(f"no Case_* directories found in {roots}")
        self._all_cases: list[EMIDECCase] = [EMIDECCase(str(d)) for d in case_dirs]

        self.store = RunStore()
        discovered = self.store.select(list(run_ids)) if run_ids else self.store.discover()
        self._runs = discovered
        self._pred_indexes = {}
        self._metrics_tables: dict[str, object] = {}
        self._metric_rows: dict[tuple[str, str], dict[int, dict]] = {}
        self._score_cache: dict[tuple[str, str], dict] = {}

        for run in discovered:
            if self.use_predictions:
                index = run.prediction_index()
                if index is not None and len(index.available()):
                    self._pred_indexes[run.run_id] = index
            table = run.load_metrics()
            if table is not None and len(table):
                self._metrics_tables[run.run_id] = table
                for row in table.rows:
                    cid = row.get("case_id")
                    lbl = row.get("class_id")
                    if cid is not None and lbl is not None:
                        self._metric_rows.setdefault((run.run_id, cid), {})[int(lbl)] = row

        self.prob_sources: "OrderedDict[str, ProbIndex]" = OrderedDict()
        self._discover_prob_sources()

        self.cases: list[EMIDECCase] = list(self._all_cases)
        self.case_pos = 0
        self.slice_idx = 0
        self._queue: list[str] = []
        self._case_cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._pred_cache: "OrderedDict[tuple[str, str], np.ndarray | None]" = OrderedDict()
        self._prob_cache: "OrderedDict[tuple[str, str], np.ndarray]" = OrderedDict()
        self._channel_cache: "OrderedDict[tuple[str, str], np.ndarray]" = OrderedDict()
        self._confusion_cache: dict[tuple[str, str, int], dict] = {}
        self._gate_cache: dict[tuple[str, str], dict | None] = {}

    # ------------------------------------------------------------------- runs
    @property
    def runs(self) -> list:
        return self._runs

    def run_choices(self, only_preds: bool = False) -> list[RunChoice]:
        out = []
        for run in self._runs:
            if only_preds and run.run_id not in self._pred_indexes:
                continue
            out.append(RunChoice(
                run_id=run.run_id, name=run.name, stage=run.manifest.stage or "?",
                has_metrics=run.run_id in self._metrics_tables))
        return out

    def run_name(self, run_id: str | None) -> str:
        if not run_id:
            return "GT only"
        for choice in self.run_choices():
            if choice.run_id == run_id:
                return choice.name
        return run_id

    def metric_run_ids(self) -> list[str]:
        return [r.run_id for r in self._runs if r.run_id in self._metrics_tables]

    def pred_run_ids(self) -> list[str]:
        return [r.run_id for r in self._runs if r.run_id in self._pred_indexes]

    def default_pred_run(self) -> str | None:
        """Prefer the best-checkpoint cascade when present."""
        preferred = (
            "cascade_503_resenc",
            "503_best_pp",
            "nnUNet_validation_500over_pp_grow1",
            "nnUNet_validation_pp_grow1",
            "nnUNet_validation_500resenc_pp_grow1",
            "nnUNet_validation_501_pp_grow1",
            "nnUNet_validation_502_pp_grow1",
            "polar_preds_erase",
        )
        available = self.pred_run_ids()
        for run_id in preferred:
            if run_id in available:
                return run_id
        return available[0] if available else None

    # ----------------------------------------------------------------- probed
    def _discover_prob_sources(self) -> None:
        """Find nnU-Net probability directories (OOF softmax .npz files)."""
        catalog = canonical_catalog(self.store.repo_root)
        if catalog is not None:
            # The viewer overlays full images. Crop-space cascade probabilities
            # need restoration before they can be offered here.
            candidates = {}
            for output in catalog["outputs"].values():
                value = output.get("paths", {}).get("probabilities")
                if value and output.get("inference", {}).get("probability_space") == "native":
                    candidates[self.store.artifact_roots.resolve(value)] = (
                        self.store.artifact_roots.resolve(output["data"]["splits_path"]))
        else:
            return
        for path, splits in candidates.items():
            if not any(path.glob("fold_*/validation/*.npz")):
                continue
            try:
                index = ProbIndex(str(path), splits)
            except Exception:
                continue
            if index.available():
                label = path.name.removeprefix("probs_")
                self.prob_sources[label] = index

    def default_prob_source(self, run_id: str | None = None) -> str | None:
        if not self.prob_sources:
            return None
        for label in self.prob_sources:
            if "probabilisticOversampling" in label:
                return label
        if run_id:
            run = next((r for r in self._runs if r.run_id == run_id), None)
            if run is not None:
                trainer = (run.manifest.get("model.trainer") or "")
                plans = (run.manifest.get("model.plans") or "")
                for label in self.prob_sources:
                    if (trainer and trainer in label) or (plans and plans in label):
                        return label
        # Prefer the richer/oversampling source when both exist.
        return list(self.prob_sources)[-1]

    def probability(self, case_id: str, source: str, label: int) -> np.ndarray | None:
        index = self.prob_sources.get(source)
        if index is None or case_id not in index.available():
            return None
        key = (source, case_id)
        if key not in self._prob_cache:
            self._prob_cache[key] = index.load(case_id)
            self._trim(self._prob_cache, 6)
        volume = self._prob_cache[key]
        return volume[label]

    # ----------------------------------------------------------------- cases
    @property
    def current_case(self) -> EMIDECCase:
        return self.cases[self.case_pos]

    def case_count(self) -> int:
        return len(self.cases)

    def set_case_view(self, filter_name: str = "All cases",
                      sort_name: str = "Case ID", run_id: str | None = None,
                      label: int = LABEL_SCAR) -> None:
        """Apply a filter/sort while keeping the current case selected."""
        current = self.current_case.case_id if self.cases else None
        cases = list(self._all_cases)
        if filter_name == "Pathological":
            cases = [c for c in cases if c.group == "Pathological"]
        elif filter_name == "Normal":
            cases = [c for c in cases if c.group == "Normal"]
        elif filter_name == "Has scar":
            cases = [c for c in cases if c.has_scar]
        elif filter_name == "Has MVO":
            cases = [c for c in cases if c.has_mvo]

        if sort_name == "GT scar (high)":
            cases.sort(key=lambda c: c.voxel_counts.get(LABEL_SCAR, 0), reverse=True)
        elif sort_name == "GT scar (low)":
            cases.sort(key=lambda c: c.voxel_counts.get(LABEL_SCAR, 0))
        else:
            cases.sort(key=lambda c: c.case_id)

        self.cases = cases
        self.case_pos = 0
        if current is not None:
            for i, case in enumerate(cases):
                if case.case_id == current:
                    self.case_pos = i
                    break
        self._clamp_case()

    def _clamp_case(self) -> None:
        if not self.cases:
            self.case_pos = 0
            return
        self.case_pos %= len(self.cases)
        self.slice_idx = min(self.slice_idx, self.slice_count() - 1)

    def set_case_index(self, index: int, reset_slice: bool = True) -> None:
        if not self.cases:
            return
        self.case_pos = index % len(self.cases)
        if reset_slice:
            self.slice_idx = self.default_slice(self.current_case)
        self._clamp_case()

    def step_case(self, delta: int) -> None:
        if self.cases:
            self.set_case_index(self.case_pos + delta)

    def case_by_id(self, case_id: str) -> "EMIDECCase | None":
        for case in self._all_cases:
            if case.case_id == case_id:
                return case
        return None

    def find_case(self, text: str) -> bool:
        """Jump to the first case whose number matches ``text``."""
        text = re.sub(r"[^0-9]", "", text or "")
        if not text:
            return False
        wanted = text.lstrip("0") or "0"
        for i, case in enumerate(self.cases):
            num = re.search(r"(\d+)$", case.case_id)
            if num and (num.group(1).lstrip("0") or "0").startswith(wanted):
                self.set_case_index(i)
                return True
        # Search the unfiltered list, then reset the filter.
        for i, case in enumerate(self._all_cases):
            num = re.search(r"(\d+)$", case.case_id)
            if num and (num.group(1).lstrip("0") or "0").startswith(wanted):
                self.set_case_view("All cases", "Case ID")
                for j, c in enumerate(self.cases):
                    if c.case_id == case.case_id:
                        self.set_case_index(j)
                        return True
        return False

    # ----------------------------------------------------------- review queue
    def set_queue(self, case_ids: list) -> int:
        """Store a review queue; returns the current case's position or -1."""
        self._queue = list(dict.fromkeys(case_ids))
        case_id = self.current_case.case_id if self.current_case else None
        return self._queue.index(case_id) if case_id in self._queue else -1

    def queue_active(self) -> bool:
        return bool(self._queue)

    def queue_ids(self) -> list:
        return list(self._queue)

    def queue_clear(self) -> None:
        self._queue = []

    def queue_step(self, delta: int) -> bool:
        """Move along the review queue (wraps around). True if a case shown."""
        if not self._queue or self.current_case is None:
            return False
        ids = self._queue
        if self.current_case.case_id in ids:
            index = ids.index(self.current_case.case_id)
        else:
            index = (0 if delta > 0 else len(ids) - 1) - delta
        target = ids[(index + delta) % len(ids)]
        for i, case in enumerate(self.cases):
            if case.case_id == target:
                self.set_case_index(i)
                return True
        # not in the current filter: reset the view, then select
        self.set_case_view("All cases", "Case ID")
        for i, case in enumerate(self.cases):
            if case.case_id == target:
                self.set_case_index(i)
                return True
        return False

    def set_slice(self, index: int) -> None:
        self.slice_idx = max(0, min(int(index), self.slice_count() - 1))

    def step_slice(self, delta: int) -> None:
        self.set_slice(self.slice_idx + delta)

    def slice_count(self) -> int:
        case = self.current_case if self.cases else None
        if case is None:
            return 1
        return int(case.load()[0].shape[2])

    def default_slice(self, case: EMIDECCase | None = None) -> int:
        case = case or self.current_case
        _, gt = case.load()
        if gt is not None and case.has_scar:
            counts = [(gt[:, :, z] == LABEL_SCAR).sum() for z in range(gt.shape[2])]
            return int(np.argmax(counts))
        img, _ = case.load()
        counts = [(img[:, :, z] > np.percentile(img, 90)).sum()
                  for z in range(img.shape[2])]
        return int(np.argmax(counts)) if counts else 0

    def fold_of(self, case_id: str, run_id: str | None) -> int | None:
        rows = self._metric_rows.get((run_id, case_id), {}) if run_id else {}
        for row in rows.values():
            if row.get("fold") is not None:
                return row["fold"]
        return None

    # ------------------------------------------------------------------ data
    def image(self, case: EMIDECCase | None = None) -> np.ndarray:
        case = case or self.current_case
        key = case.case_id
        if key not in self._case_cache:
            self._case_cache[key] = case.load()
            self._trim(self._case_cache, self._cache_cases)
        return self._case_cache[key][0]

    def gt(self, case: EMIDECCase | None = None) -> np.ndarray | None:
        case = case or self.current_case
        key = case.case_id
        if key not in self._case_cache:
            self._case_cache[key] = case.load()
            self._trim(self._case_cache, self._cache_cases)
        return self._case_cache[key][1]

    def channel(self, case_id: str, channel_name: str) -> np.ndarray:
        kind = CHANNELS.get(channel_name, "lge")
        key = (case_id, kind)
        if key not in self._channel_cache:
            case = next((c for c in self._all_cases if c.case_id == case_id), None)
            if case is None:
                return np.zeros((1, 1, 1), dtype=np.float32)
            img, _ = case.load()
            self._channel_cache[key] = derived_channel(img, kind)
            self._trim(self._channel_cache, 6)
        return self._channel_cache[key]

    def prediction(self, case_id: str, run_id: str | None) -> np.ndarray | None:
        if not run_id:
            return None
        index = self._pred_indexes.get(run_id)
        if index is None or case_id not in index.available():
            return None
        key = (run_id, case_id)
        if key not in self._pred_cache:
            self._pred_cache[key] = index.load(case_id).astype(np.uint8)
            self._trim(self._pred_cache, self._cache_preds)
        return self._pred_cache[key]

    # --------------------------------------------------------------- metrics
    def case_scores(self, case_id: str, run_id: str | None) -> dict[int, dict]:
        """Per-class metrics for one case/run.

        Prefers saved metrics (which carry HD95/TP/FP/FN); falls back to
        computing Dice and volumes from the prediction file.
        """
        if not run_id:
            return {}
        key = (run_id, case_id)
        if key in self._score_cache:
            return self._score_cache[key]
        rows = {lbl: dict(row) for lbl, row in
                self._metric_rows.get(key, {}).items()}
        pred = self.prediction(case_id, run_id)
        case = next((c for c in self._all_cases if c.case_id == case_id), None)
        gt = None
        if pred is not None and case is not None:
            _, gt = case.load()
        if rows:
            # v1 metrics lack TP/FP/FN; fill them from the arrays so the HUD can
            # still show recall/precision without re-scoring.
            if pred is not None and gt is not None and pred.shape == gt.shape:
                for label in EVAL_LABELS:
                    row = rows.get(label)
                    if row is None or row.get("tp") is not None:
                        continue
                    g = gt == label
                    p = pred == label
                    row["tp"] = int((g & p).sum())
                    row["fp"] = int((~g & p).sum())
                    row["fn"] = int((g & ~p).sum())
                    if row.get("gt_voxels") is None:
                        row["gt_voxels"] = int(g.sum())
                    if row.get("pred_voxels") is None:
                        row["pred_voxels"] = int(p.sum())
        if not rows:
            if pred is None or case is None:
                self._score_cache[key] = {}
                return {}
            if gt is None:
                self._score_cache[key] = {}
                return {}
            for label in EVAL_LABELS:
                if gt is None or pred.shape != gt.shape:
                    continue
                g = gt == label
                p = pred == label
                tp = int((g & p).sum())
                fp = int((~g & p).sum())
                fn = int((g & ~p).sum())
                denom = 2 * tp + fp + fn
                dice = 1.0 if denom == 0 else 2 * tp / denom
                rows[label] = {
                    "class_id": label, "dice": dice, "dice_defined": True,
                    "hd95": None, "hd95_defined": False,
                    "gt_voxels": int(g.sum()), "pred_voxels": int(p.sum()),
                    "tp": tp, "fp": fp, "fn": fn,
                }
        self._score_cache[key] = rows
        return rows

    def confusion(self, case_id: str, run_id: str | None,
                  label: int) -> dict[str, float]:
        """Distribution of predicted labels inside GT ``label`` voxels."""
        if not run_id:
            return {}
        key = (run_id, case_id, int(label))
        if key in self._confusion_cache:
            return self._confusion_cache[key]
        case = next((c for c in self._all_cases if c.case_id == case_id), None)
        pred = self.prediction(case_id, run_id)
        if case is None or pred is None:
            return {}
        _, gt = case.load()
        if gt is None or pred.shape != gt.shape:
            return {}
        mask = gt == label
        total = int(mask.sum())
        if total == 0:
            return {}
        counts = np.bincount(pred[mask].astype(np.int64), minlength=5)
        out = {LABEL_NAMES[i]: float(counts[i]) / total for i in range(5)}
        self._confusion_cache[key] = out
        return out

    # ------------------------------------------------------------ gate preview
    def gate_preview(self, case_id: str, run_id: str | None) -> dict | None:
        """What do the challenge gates change on this prediction?

        Applies the locked postprocessing rules (scar size gate, optional
        in-plane scar growth, MVO size gate, MVO-in-scar containment) to the
        run's prediction and returns per-rule removal counts plus before/after
        Dice. ``removed`` is a boolean mask for canvas marks. Falls back to
        None when no prediction/GT is available.
        """
        key = (run_id or "", case_id)
        if key in self._gate_cache:
            return self._gate_cache[key]
        pred = self.prediction(case_id, run_id)
        case = self.case_by_id(case_id)
        gt = None
        if case is not None:
            _, gt = case.load()
        info = None
        if pred is not None and gt is not None and pred.shape == gt.shape:
            from src.datautils.postprocess import PostProcessor, MVO_LABEL, SCAR_LABEL

            grow_px = int(re.search(r"grow([1-9])", run_id or "").group(1)) \
                if re.search(r"grow[1-9]", run_id or "") else 0
            post = PostProcessor(scar_grow_px=grow_px)
            out = post(pred)
            removed = (pred != 0) & (out == 0)

            def _counts(label: int) -> dict:
                mask = (pred == label) & removed
                _, n = ndi.label(mask)
                return {"voxels": int(mask.sum()), "comps": int(n)}

            info = {
                "changed": bool((out != pred).any()),
                "grow_px": grow_px,
                "scar": _counts(SCAR_LABEL),
                "mvo": _counts(MVO_LABEL),
                "removed": removed,
                "before": self._plain_dice(pred, gt),
                "after": self._plain_dice(out, gt),
            }
        self._gate_cache[key] = info
        return info

    @staticmethod
    def _plain_dice(pred: np.ndarray, gt: np.ndarray) -> dict:
        """Per-label Dice with the both-empty-is-1.0 convention, plus mean."""
        dices = {}
        for label in (1, 2, 3, 4):
            g = gt == label
            p = pred == label
            denom = int(g.sum()) + int(p.sum())
            dices[label] = 1.0 if denom == 0 else                 2.0 * int((g & p).sum()) / denom
        dices["all"] = sum(dices.values()) / 4.0
        return dices

    def metric_values(self, run_id: str, label: int, metric: str = "dice",
                      group: str | None = None) -> list[float]:
        table = self._metrics_tables.get(run_id)
        if table is None:
            return []
        defined = "dice_defined" if metric == "dice" else "hd95_defined"
        vals = []
        for row in table.rows:
            if row.get("class_id") != label or not row.get(defined):
                continue
            if group and group != "All" and row.get("group") != group:
                continue
            value = row.get(metric)
            if value is not None and np.isfinite(value):
                vals.append(float(value))
        return vals

    def run_table(self, run_id: str):
        return self._metrics_tables.get(run_id)

    def bad_cases(self, run_id: str, threshold: float = 0.2,
                  group: str | None = None) -> list[dict]:
        """Cases below ``threshold`` on any class for one run."""
        table = self._metrics_tables.get(run_id)
        if table is None:
            return []
        by_case: dict[str, dict] = {}
        for row in table.rows:
            cid = row.get("case_id")
            lbl = row.get("class_id")
            if cid is None or lbl is None:
                continue
            if group and group != "All" and row.get("group") != group:
                continue
            entry = by_case.setdefault(cid, {
                "case_id": cid, "group": row.get("group"),
                "fold": row.get("fold"), "dice": {}})
            if row.get("dice_defined"):
                entry["dice"][int(lbl)] = row.get("dice")
        out = []
        for entry in by_case.values():
            vals = [v for v in entry["dice"].values() if v is not None]
            if vals and min(vals) < threshold:
                out.append(entry)
        out.sort(key=lambda e: min(
            [v for v in e["dice"].values() if v is not None] or [1.0]))
        return out

    # ------------------------------------------------------------------ misc
    @staticmethod
    def group_of(case_id: str) -> str:
        if case_id.startswith("Case_P"):
            return "Pathological"
        if case_id.startswith("Case_N"):
            return "Normal"
        return "Unknown"

    @staticmethod
    def _trim(cache: "OrderedDict", limit: int) -> None:
        while len(cache) > max(1, limit):
            cache.popitem(last=False)

    def describe(self) -> str:
        return (f"{len(self._all_cases)} cases · "
                f"{len(self._pred_indexes)} prediction runs · "
                f"{len(self._metrics_tables)} metric tables")
