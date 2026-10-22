"""Segmentation metrics: Dice, HD95 (surface-distance lib), reliability curves."""

import numpy as np
from scipy.spatial.distance import dice as dice_dissimilarity
from surface_distance import compute_robust_hausdorff, compute_surface_distances

DEFAULT_LABELS = (1, 2, 3, 4)


class SegmentationMetrics:
    """Per-class segmentation metrics with empty-class handling.

    - Dice: both empty -> 1.0, one empty -> 0.0.
    - HD95: empty on either side -> NaN (caller decides aggregation).
    - Reliability: vectorized numpy binning (keeps empty bins as NaN, with
      counts — sklearn's calibration_curve drops them, so numpy is used).
    """

    def __init__(self, labels: tuple[int, ...] = DEFAULT_LABELS):
        self.labels = tuple(labels)

    @staticmethod
    def _mask(vol: np.ndarray, label: int) -> np.ndarray:
        return np.asarray(vol == label, dtype=bool)

    def dice(self, pred: np.ndarray, gt: np.ndarray) -> dict[int, float]:
        """Per-class Dice via ``scipy.spatial.distance.dice``.

        Both empty -> 1.0 (scipy yields NaN there, so it is short-circuited).
        """
        scores = {}
        for lbl in self.labels:
            p, g = self._mask(pred, lbl), self._mask(gt, lbl)
            if not p.any() and not g.any():
                scores[lbl] = 1.0
                continue
            scores[lbl] = float(1.0 - dice_dissimilarity(p.ravel(), g.ravel()))
        return scores

    def confusion(self, pred: np.ndarray, gt: np.ndarray,
                  n_labels: int = 5) -> np.ndarray:
        """Confusion matrix ``C[gt_label, pred_label]`` over ``n_labels`` labels.

        One matrix per case replaces separate TP/FP/FN counters: they are
        projections of it, and it also shows *where* errors went.
        """
        gt = np.asarray(gt).ravel()
        pred = np.asarray(pred).ravel()
        if gt.shape != pred.shape:
            raise ValueError(f"pred/gt size mismatch: {pred.shape} vs {gt.shape}")
        if gt.size and (gt.min() < 0 or gt.max() >= n_labels):
            raise ValueError(f"gt labels outside [0, {n_labels})")
        if pred.size and (pred.min() < 0 or pred.max() >= n_labels):
            raise ValueError(f"pred labels outside [0, {n_labels})")
        idx = gt.astype(np.int64) * n_labels + pred.astype(np.int64)
        return np.bincount(idx, minlength=n_labels ** 2).reshape(n_labels, n_labels)

    def hd95(self, pred: np.ndarray, gt: np.ndarray,
             spacing: tuple[float, ...]) -> dict[int, float]:
        scores = {}
        for lbl in self.labels:
            p, g = self._mask(pred, lbl), self._mask(gt, lbl)
            if not p.any() or not g.any():
                scores[lbl] = float("nan")
                continue
            surfaces = compute_surface_distances(
                g, p, spacing_mm=tuple(float(s) for s in spacing))
            scores[lbl] = float(compute_robust_hausdorff(surfaces, 95))
        return scores

    @staticmethod
    def reliability(probs: np.ndarray, gt_foreground: np.ndarray,
                    n_bins: int = 10) -> dict[str, np.ndarray]:
        """Bin predicted P(foreground) vs observed foreground rate."""
        probs = np.asarray(probs, dtype=float).ravel()
        gt = np.asarray(gt_foreground, dtype=bool).ravel()
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        # Bin 0 is [0, edge] inclusive of 0; others are (lo, hi].
        idx = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins)
        conf_sum = np.bincount(idx, weights=probs, minlength=n_bins)
        obs_sum = np.bincount(idx, weights=gt.astype(float), minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_conf = np.where(counts, conf_sum / np.maximum(counts, 1), np.nan)
            observed = np.where(counts, obs_sum / np.maximum(counts, 1), np.nan)
        return {
            "bin_centers": 0.5 * (edges[:-1] + edges[1:]),
            "mean_confidence": mean_conf,
            "observed_rate": observed,
            "counts": counts,
        }

    def evaluate(self, pred: np.ndarray, gt: np.ndarray,
                 spacing: tuple[float, ...]) -> dict[str, dict[int, float]]:
        """Bundle Dice + HD95 for one case."""
        return {"dice": self.dice(pred, gt),
                "hd95": self.hd95(pred, gt, spacing)}
