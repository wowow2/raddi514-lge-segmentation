import os
import glob
import re
import nibabel as nib
import numpy as np

def parse_metadata_file(filepath):
    """Parses EMIDEC companion text files into a key-value dictionary."""
    if not os.path.exists(filepath):
        return {}
    meta = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Case"):
                continue
            parts = re.split(r"[:\t]", line, maxsplit=1)
            if len(parts) == 2:
                k = parts[0].strip().replace("", "")
                v = parts[1].strip().replace("", "")
                meta[k] = v
    return meta

class EMIDECCase:
    def __init__(self, case_dir):
        self.case_dir = case_dir
        self.case_id = os.path.basename(case_dir)
        self.group = "Pathological" if self.case_id.startswith("Case_P") else "Normal"
        
        self.img_path = os.path.join(case_dir, "Images", f"{self.case_id}.nii.gz")
        self.gt_path  = os.path.join(case_dir, "Contours", f"{self.case_id}.nii.gz")
        
        # Discover metadata file (handles 'Case_P060.txt', 'Case P060.txt', or inside case folder)
        parent_dir = os.path.dirname(case_dir)
        raw_num = self.case_id.replace("Case_", "")
        txt_candidates = (
            glob.glob(os.path.join(parent_dir, f"*{raw_num}*.txt")) +
            glob.glob(os.path.join(case_dir, "*.txt"))
        )
        self.meta_path = txt_candidates[0] if txt_candidates else None
        self.clinical_meta = parse_metadata_file(self.meta_path) if self.meta_path else {}
        
        # Lazy loaded arrays
        self._img = None
        self._gt = None
        self._header = None
        self.zooms = None
        self.shape = None
        
    def load(self):
        if self._img is None:
            nii = nib.load(self.img_path)
            self._header = nii.header
            self.zooms = [round(float(z), 2) for z in nii.header.get_zooms()]
            
            raw_img = nii.get_fdata(dtype=np.float32)
            # Volume percentile normalization for display
            vmin, vmax = np.percentile(raw_img, 1), np.percentile(raw_img, 99)
            self._img = np.clip((raw_img - vmin) / max(vmax - vmin, 1e-5), 0, 1)
            self.shape = self._img.shape
            
            gt_nii = nib.load(self.gt_path)
            self._gt = gt_nii.get_fdata().astype(np.uint8)
        return self._img, self._gt

    @property
    def has_scar(self):
        _, gt = self.load()
        return bool((gt == 3).any())

    @property
    def has_mvo(self):
        _, gt = self.load()
        return bool((gt == 4).any())

    @property
    def voxel_counts(self):
        _, gt = self.load()
        return {lbl: int((gt == lbl).sum()) for lbl in range(5)}
