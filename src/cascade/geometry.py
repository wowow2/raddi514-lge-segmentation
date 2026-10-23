"""ROI geometry for the LGE stage-2 cascade."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class CropBox:
    """In-plane crop; Z is never cropped."""

    x0: int
    x1: int
    y0: int
    y1: int

    def __post_init__(self) -> None:
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError(f"negative crop origin: {self}")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"empty crop: {self}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_dict(self) -> dict:
        return {
            "x0": self.x0, "x1": self.x1,
            "y0": self.y0, "y1": self.y1,
            "width": self.width, "height": self.height,
        }


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 3D component, ignoring distant stage-1 false positives."""
    labelled, count = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return mask
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return labelled == int(sizes.argmax())


def foreground_bbox(mask: np.ndarray, pad: int = 16, min_size: int = 64,
                    multiple: int = 16) -> CropBox:
    """Square, padded, size-quantised in-plane box around foreground voxels."""
    if mask.ndim != 3:
        raise ValueError(f"mask must be (X, Y, Z), got {mask.shape}")
    if pad < 0 or min_size < 1 or multiple < 1:
        raise ValueError("pad must be >= 0; min_size and multiple must be >= 1")

    xs, ys, _ = np.nonzero(mask)
    size_x, size_y = mask.shape[:2]
    max_side = min(size_x, size_y)

    if xs.size == 0:
        side = min(max(min_size, 1), max_side)
        x0 = (size_x - side) // 2
        y0 = (size_y - side) // 2
        return CropBox(x0, x0 + side, y0, y0 + side)

    side = max(int(xs.max() - xs.min() + 1),
               int(ys.max() - ys.min() + 1)) + 2 * pad
    side = max(side, min_size)
    side = min(int(np.ceil(side / multiple) * multiple), max_side)
    if side % multiple:
        side = max(side - side % multiple, min(multiple, max_side))

    cx = 0.5 * (int(xs.min()) + int(xs.max()))
    cy = 0.5 * (int(ys.min()) + int(ys.max()))
    x0 = min(max(int(round(cx - side / 2)), 0), size_x - side)
    y0 = min(max(int(round(cy - side / 2)), 0), size_y - side)
    return CropBox(x0, x0 + side, y0, y0 + side)


def crop_array(array: np.ndarray, box: CropBox) -> np.ndarray:
    """Crop an (X, Y, Z) or (C, X, Y, Z) array in-plane."""
    if array.ndim == 3:
        return array[box.x0:box.x1, box.y0:box.y1, :]
    if array.ndim == 4:
        return array[:, box.x0:box.x1, box.y0:box.y1, :]
    raise ValueError(f"expected 3D or 4D array, got {array.shape}")


def shifted_affine(affine: np.ndarray, box: CropBox) -> np.ndarray:
    """Move the affine origin to the crop's upper-left voxel."""
    out = np.array(affine, dtype=float, copy=True)
    out[:3, 3] += out[:3, 0] * box.x0 + out[:3, 1] * box.y0
    return out
