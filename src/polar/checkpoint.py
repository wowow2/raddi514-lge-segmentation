"""Checkpoint helpers for the polar stage-2 model.

New checkpoints store a full ``config`` block (model, data, training) so a
model can be rebuilt without guessing flags. Legacy checkpoints only stored
``use_radius`` / ``deep_supervision``; those are still understood.
"""

from pathlib import Path

import torch

from .model import PolarUNet

DEFAULT_MODEL_PARAMS = {
    "in_channels": 1,
    "num_classes": 5,
    "use_radius": False,
    "deep_supervision": True,
}


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(str(path), map_location=map_location)


def checkpoint_config(ckpt: dict) -> dict:
    """Return a normalised config dict, falling back to legacy top-level flags."""
    config = dict(ckpt.get("config") or {})
    model = dict(config.get("model") or {})
    params = dict(model.get("params") or {})
    params.setdefault("use_radius", bool(ckpt.get("use_radius", False)))
    params.setdefault("deep_supervision", bool(ckpt.get("deep_supervision", True)))
    for key, value in DEFAULT_MODEL_PARAMS.items():
        params.setdefault(key, value)
    config["model"] = {"name": model.get("name", "polar_unet"), "params": params}
    return config


def build_model(config: dict, model_cls=PolarUNet) -> torch.nn.Module:
    params = (config.get("model") or {}).get("params") or {}
    return model_cls(**params)


def load_model(path: str | Path, device: str = "cpu",
               model_cls=PolarUNet) -> tuple[torch.nn.Module, dict]:
    """Load model weights and config from ``checkpoint_best.pt`` / ``.pth``."""
    ckpt = load_checkpoint(path, map_location=device)
    config = checkpoint_config(ckpt)
    model = build_model(config, model_cls=model_cls)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, config


def polar_resolution(config: dict) -> tuple[int, int]:
    data = config.get("data") or {}
    return int(data.get("R", 64)), int(data.get("Theta", 256))
