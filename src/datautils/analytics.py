"""Dataset-wide analytics for EMIDEC. Pure compute — zero matplotlib.

Results are JSON-serializable (lists, not ndarrays) and cached to disk so the
viewer dashboard and report scripts load instantly after the first run.
Cache files are keyed by case set (see cache_path_for) so different splits
never share a stale cache.
"""

import hashlib
import json
import os

import numpy as np

from .emidec_loader import LABEL_CAVITY, LABEL_MVO, LABEL_MYOCARDIUM, LABEL_SCAR

DEFAULT_CACHE_PATH = "results/analytics_cache.json"
INTENSITY_SAMPLES_PER_CLASS = 2000
N_SLICE_BINS = 10


def _num(value) -> float | None:
    """Parse a clinical metadata string to float ('4,5' -> 4.5, 'N/A' -> None)."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


class DatasetAnalytics:
    """Aggregate statistics over EMIDEC cases.

    Usage:
        stats = DatasetAnalytics(cases).compute()   # dict of analyses
        DatasetAnalytics(cases).save()              # -> results/analytics_cache.json
        stats = DatasetAnalytics.load()             # cached dict (no data access)
    """

    def __init__(self, cases, cache_path: str | None = None,
                 intensity_samples: int = INTENSITY_SAMPLES_PER_CLASS):
        self.cases = list(cases)
        self.cache_path = cache_path or self.cache_path_for(self.cases)
        self.intensity_samples = intensity_samples
        self.stats: dict | None = None

    @staticmethod
    def cache_path_for(cases, base_dir: str = "results") -> str:
        """Cache path keyed by case set (first 8 hex of md5 over sorted ids)."""
        ids = sorted(c.case_id for c in cases)
        digest = hashlib.md5(",".join(ids).encode()).hexdigest()[:8]
        return os.path.join(base_dir, f"analytics_{digest}.json")

    def compute(self) -> dict:
        """Stream all cases once; return the analyses dict (also on self.stats)."""
        rng = np.random.default_rng(42)
        intensities = {lbl: [] for lbl in
                       (LABEL_CAVITY, LABEL_MYOCARDIUM, LABEL_SCAR, LABEL_MVO)}
        per_case = []
        slice_hits = np.zeros(N_SLICE_BINS, dtype=int)
        slice_totals = np.zeros(N_SLICE_BINS, dtype=int)

        for case in self.cases:
            img, gt = case.load()
            vox = case.voxel_counts
            myo = vox[LABEL_MYOCARDIUM] + vox[LABEL_SCAR] + vox[LABEL_MVO]
            scar = vox[LABEL_SCAR] + vox[LABEL_MVO]
            meta = case.clinical_meta
            per_case.append({
                "case_id": case.case_id,
                "group": case.group,
                "has_gt": case.has_gt,
                "has_scar": case.has_scar,
                "has_mvo": case.has_mvo,
                "scar_voxels": scar,
                "scar_pct_myo": 100.0 * scar / myo if myo else 0.0,
                "n_slices": img.shape[2],
                "spacing_xy": case.zooms[:2],
                "shape_xy": list(img.shape[:2]),
                "age": _num(meta.get("Age")),
                "lvef": _num(meta.get("FEVG")),
                "troponin": _num(meta.get("Troponin")),
                "killip": _num(meta.get("Killip Max")),
            })
            if case.has_gt:
                for lbl, buf in intensities.items():
                    vals = img[gt == lbl]
                    if vals.size:
                        n = min(vals.size, self.intensity_samples)
                        buf.extend(rng.choice(vals, n, replace=False).tolist())
                z = img.shape[2]
                for s in range(z):
                    b = min(int(s / z * N_SLICE_BINS), N_SLICE_BINS - 1)
                    slice_totals[b] += 1
                    if ((gt[:, :, s] == LABEL_SCAR).any()
                            or (gt[:, :, s] == LABEL_MVO).any()):
                        slice_hits[b] += 1

        total_vox = {str(lbl): 0 for lbl in range(5)}
        for case in self.cases:
            if case.has_gt:
                for lbl, n in case.voxel_counts.items():
                    total_vox[str(lbl)] += n

        self.stats = {
            "n_cases": len(self.cases),
            "voxel_counts": total_vox,
            "intensities": {str(k): v for k, v in intensities.items()},
            "per_case": per_case,
            "slice_involvement": {
                "bin_edges": np.linspace(0.0, 1.0, N_SLICE_BINS + 1).tolist(),
                "scar_slices": slice_hits.tolist(),
                "total_slices": slice_totals.tolist(),
            },
        }
        return self.stats

    def save(self, path: str | None = None) -> str:
        """Persist stats to JSON. Returns the path written."""
        path = path or self.cache_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.stats if self.stats is not None else self.compute(),
                      f)
        return path

    @classmethod
    def load(cls, path: str = DEFAULT_CACHE_PATH) -> dict:
        """Load cached stats (no dataset access)."""
        with open(path) as f:
            return json.load(f)

    def __repr__(self) -> str:
        return f"DatasetAnalytics(n_cases={len(self.cases)})"
