import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .emidec_loader import EMIDECCase, N_LABELS


class EMIDECDataset(Dataset):
    """A collection of EMIDEC cases, usable directly with torch DataLoaders.

    ``__getitem__`` returns ``(image, ground_truth)`` torch tensors with
    shape (1, X, Y, Z), dtypes float32 / int64 respectively. The mask is
    None for blind cases without ground truth.
    """

    def __init__(self, root_dir: str | list[str], group: str | None = None):
        """Args:
            root_dir: directory (or list of directories) containing ``Case_*``
                subfolders (e.g. train/).
            group: optionally restrict to "Pathological" or "Normal".
        """
        roots = [root_dir] if isinstance(root_dir, str) else list(root_dir)
        # Sorted per root so multi-root order follows the caller (train, then test).
        case_dirs = [
            d
            for root in roots
            for d in sorted(
                dd for dd in glob.glob(os.path.join(root, "Case_*")) if os.path.isdir(dd)
            )
        ]
        if not case_dirs:
            raise ValueError(f"No cases found in {roots}")
        cases = [EMIDECCase(d) for d in case_dirs]
        if group is not None:
            cases = [c for c in cases if c.group == group]
            if not cases:
                raise ValueError(f"No '{group}' cases found in {root_dir}")
        self.root_dir = root_dir
        self.cases: list[EMIDECCase] = cases
        self._by_id = {c.case_id: c for c in cases}

    @classmethod
    def from_dir(cls, root_dir: str | list[str], group: str | None = None) -> "EMIDECDataset":
        return cls(root_dir, group=group)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        case = self.cases[idx]
        img, gt = case.load()
        image = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float()
        mask = (
            torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0).long()
            if gt is not None
            else None  # blind case: no ground truth available
        )
        return image, mask

    def __repr__(self) -> str:
        return f"EMIDECDataset(root={self.root_dir!r}, n={len(self)})"

    def get(self, case_id: str) -> EMIDECCase:
        """Return the case with the given id (KeyError if absent)."""
        return self._by_id[case_id]

    @property
    def pathological(self) -> list[EMIDECCase]:
        return [c for c in self.cases if c.group == "Pathological"]

    @property
    def normal(self) -> list[EMIDECCase]:
        return [c for c in self.cases if c.group == "Normal"]

    @property
    def case_ids(self) -> list[str]:
        return [c.case_id for c in self.cases]

    def summary(self) -> str:
        """Human-readable overview: group counts + aggregate voxel distribution.

        Voxel stats aggregate only cases with ground truth; blind cases are
        reported separately so they don't skew the percentages.
        """
        n_path = len(self.pathological)
        n_norm = len(self.normal)
        n_unknown = len(self) - n_path - n_norm
        gt_cases = [c for c in self.cases if c.has_gt]
        n_blind = len(self.cases) - len(gt_cases)
        total_vox = {lbl: 0 for lbl in range(N_LABELS)}
        for case in gt_cases:
            for lbl, count in case.voxel_counts.items():
                total_vox[lbl] += count
        grand = sum(total_vox.values()) or 1
        lines = [
            f"EMIDECDataset: {len(self)} cases "
            f"({n_path} Pathological, {n_norm} Normal"
            f"{f', {n_unknown} Unknown' if n_unknown else ''}) @ {self.root_dir}",
        ]
        if n_blind:
            lines.append(f"  ({n_blind} blind cases without ground truth)")
        lines.append("Aggregate voxel distribution (GT cases only):")
        names = {0: "Background", 1: "Cavity", 2: "Myocardium", 3: "Scar", 4: "MVO"}
        for lbl in range(N_LABELS):
            pct = 100.0 * total_vox[lbl] / grand
            lines.append(
                f"  {lbl} {names[lbl]:<11}: {total_vox[lbl]:>12,d} voxels ({pct:5.2f}%)"
            )
        report = "\n".join(lines)
        print(report)
        return report
