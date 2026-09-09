"""Loading utilities for a single EMIDEC case (LGE cardiac MRI + contour masks)."""

import glob
import os
import re
import unicodedata

import nibabel as nib
import numpy as np

# Segmentation label ids in the ground-truth contours.
LABEL_BACKGROUND = 0
LABEL_CAVITY = 1
LABEL_MYOCARDIUM = 2
LABEL_SCAR = 3
LABEL_MVO = 4
N_LABELS = 5


def _clean_text(text: str) -> str:
    """Strip non-ASCII/junk characters left by the source text-file encoding."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


def parse_metadata_file(filepath: str | None) -> dict[str, str]:
    """Parse an EMIDEC companion text file into a key-value dictionary.

    Lines are ``Key : Value`` (``:`` or tab separated); the leading
    ``Case ...`` header line is skipped. Returns {} when the file is missing.
    """
    if not filepath or not os.path.exists(filepath):
        return {}
    meta: dict[str, str] = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Case"):
                continue
            parts = re.split(r"[:\t]", line, maxsplit=1)
            if len(parts) == 2:
                key = _clean_text(parts[0])
                value = _clean_text(parts[1])
                if key:
                    meta[key] = value
    return meta


def discover_metadata_path(case_dir: str, case_id: str) -> str | None:
    """Find the companion .txt metadata file for a case.

    Handles 'Case_P060.txt', 'Case P060.txt', or a file inside the case folder.
    """
    parent_dir = os.path.dirname(case_dir)
    raw_num = case_id.replace("Case_", "")
    candidates = glob.glob(os.path.join(parent_dir, f"*{raw_num}*.txt"))
    candidates += glob.glob(os.path.join(case_dir, "*.txt"))
    return candidates[0] if candidates else None


class EMIDECCase:
    """One EMIDEC patient case: LGE image, ground-truth contours, clinical metadata.

    Image/GT arrays are loaded lazily on first access and cached.
    """

    def __init__(self, case_dir: str):
        self.case_dir = case_dir
        self.case_id = os.path.basename(case_dir)
        if self.case_id.startswith("Case_P"):
            self.group = "Pathological"
        elif self.case_id.startswith("Case_N"):
            self.group = "Normal"
        else:  # blind test ids look like Case_101
            self.group = "Unknown"

        self.img_path = os.path.join(case_dir, "Images", f"{self.case_id}.nii.gz")
        self.gt_path = os.path.join(case_dir, "Contours", f"{self.case_id}.nii.gz")
        # Blind test cases ship no Contours/ folder.
        self.has_gt = os.path.exists(self.gt_path)

        self.meta_path = discover_metadata_path(case_dir, self.case_id)
        self.clinical_meta = parse_metadata_file(self.meta_path)

        # Lazily loaded state.
        self._img: np.ndarray | None = None
        self._gt: np.ndarray | None = None
        self._voxel_counts: dict[int, int] | None = None
        self.zooms: list[float] | None = None
        self.shape: tuple[int, ...] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._img is not None and (self._gt is not None or not self.has_gt)

    def load(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Load and return (raw image float32, ground-truth uint8 or None).

        Arrays have shape (X, Y, Z). Ground truth is None for blind cases.
        """
        if not self.is_loaded:
            if not os.path.exists(self.img_path):
                raise FileNotFoundError(f"Missing image for {self.case_id}: {self.img_path}")

            nii = nib.load(self.img_path)
            self.zooms = [round(float(z), 2) for z in nii.header.get_zooms()]
            self._img = nii.get_fdata(dtype=np.float32)

            if self.has_gt:
                self._gt = nib.load(self.gt_path).get_fdata().astype(np.uint8)
                if self._img.shape != self._gt.shape:
                    raise ValueError(
                        f"Image/GT shape mismatch for {self.case_id}: "
                        f"{self._img.shape} vs {self._gt.shape}"
                    )
            self.shape = self._img.shape
        return self._img, self._gt

    @property
    def display_image(self) -> np.ndarray:
        """Image normalized to [0, 1] via 1st/99th percentile clipping (for display)."""
        img, _ = self.load()
        vmin, vmax = np.percentile(img, 1), np.percentile(img, 99)
        return np.clip((img - vmin) / max(vmax - vmin, 1e-5), 0, 1)

    @property
    def voxel_counts(self) -> dict[int, int]:
        """Voxel count per label (computed once, then cached; zeros if blind)."""
        if self._voxel_counts is None:
            if not self.has_gt:
                self._voxel_counts = {lbl: 0 for lbl in range(N_LABELS)}
            else:
                _, gt = self.load()
                self._voxel_counts = {lbl: int((gt == lbl).sum()) for lbl in range(N_LABELS)}
        return self._voxel_counts

    @property
    def has_scar(self) -> bool:
        return self.voxel_counts[LABEL_SCAR] > 0

    @property
    def has_mvo(self) -> bool:
        return self.voxel_counts[LABEL_MVO] > 0

    def __repr__(self) -> str:
        return f"EMIDECCase({self.case_id!r}, group={self.group!r})"
