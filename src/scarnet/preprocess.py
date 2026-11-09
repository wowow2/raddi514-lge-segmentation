"""Shared crop, normalisation, and resize helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.cascade.geometry import CropBox, foreground_bbox, largest_component


def normalize_slice(img_slice: np.ndarray,
                    clip_pct: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    lo, hi = np.percentile(img_slice, clip_pct)
    clipped = np.clip(img_slice, lo, hi)
    return (clipped - lo) / max(hi - lo, 1e-5)


def prepare_prior(pred: np.ndarray,
                  use_largest_component: bool = True) -> np.ndarray:
    mask = np.asarray(pred) > 0
    return largest_component(mask) if use_largest_component else mask


def prior_box(mask: np.ndarray, pad: int = 16, min_size: int = 64,
              multiple: int = 16) -> CropBox:
    return foreground_bbox(mask, pad=pad, min_size=min_size, multiple=multiple)


def prepare_image_slice(img_slice: np.ndarray, out_size: int,
                        clip_pct: tuple[float, float] = (1.0, 99.0)) -> torch.Tensor:
    normalized = normalize_slice(img_slice, clip_pct).astype(np.float32)
    tensor = torch.from_numpy(normalized)[None, None]
    size = (out_size, out_size)
    if tensor.shape[-2:] != size:
        tensor = F.interpolate(tensor, size=size, mode="bilinear",
                               align_corners=False, antialias=True)
    return tensor[0]


def prepare_mask_slice(mask_slice: np.ndarray, out_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(mask_slice).astype(np.float32))[None, None]
    size = (out_size, out_size)
    if tensor.shape[-2:] != size:
        tensor = F.interpolate(tensor, size=size, mode="nearest")
    return tensor[0, 0].long()
