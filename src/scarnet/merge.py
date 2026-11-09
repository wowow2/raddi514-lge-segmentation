"""Resize ScarNet predictions and paste them into full-volume space."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.cascade.geometry import CropBox


def resize_label(pred: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a 2D label map."""
    pred = np.asarray(pred)
    if pred.shape == tuple(size):
        return pred.astype(np.uint8, copy=False)
    tensor = torch.from_numpy(pred.astype(np.float32))[None, None]
    out = F.interpolate(tensor, size=tuple(size), mode="nearest")
    return out[0, 0].round().long().numpy().astype(np.uint8)


def resize_probabilities(probs: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinearly resize a ``(C, H, W)`` probability map."""
    probs = np.asarray(probs, dtype=np.float32)
    if probs.shape[-2:] == tuple(size):
        return probs
    tensor = torch.from_numpy(probs)[None]
    out = F.interpolate(tensor, size=tuple(size), mode="bilinear",
                        align_corners=False)
    return out[0].numpy()


def merge_slice_predictions(predictions: dict[int, np.ndarray],
                            box: CropBox,
                            full_shape: tuple[int, int, int],
                            fill: int = 0) -> np.ndarray:
    """Resize each ``{z: (S, S)}`` prediction and paste it into a full volume."""
    full = np.full(full_shape, fill, dtype=np.uint8)
    z_max = full_shape[2]
    for z, pred in predictions.items():
        if z < 0 or z >= z_max:
            raise IndexError(f"slice {z} outside full volume with {z_max} slices")
        resized = resize_label(pred, (box.width, box.height))
        full[box.x0:box.x1, box.y0:box.y1, z] = resized
    return full
