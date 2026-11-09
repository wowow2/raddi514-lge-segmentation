"""ScarNet checkpoint helpers."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch


def import_class(spec: str):
    """Import ``package.module:ClassName`` (or ``package.module.ClassName``)."""
    if not spec:
        raise ValueError("model class spec is empty")
    if ":" in spec:
        module_name, _, class_name = spec.partition(":")
    else:
        module_name, _, class_name = spec.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(
            f"model class must look like 'pkg.module:Class', got {spec!r}")
    return getattr(importlib.import_module(module_name), class_name)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def load_model(path: str | Path, device: str = "cpu",
               model_cls=None) -> tuple[torch.nn.Module, dict]:
    """Rebuild the recorded model class and load ``checkpoint_best.pt``."""
    ckpt = load_checkpoint(path, map_location=device)
    config = dict(ckpt.get("config") or {})
    meta = dict(config.get("model") or {})
    params = dict(meta.get("params") or {})
    params.pop("medsam_ckpt", None)
    spec = meta.get("class")
    if model_cls is not None:
        model = model_cls(**params)
    elif spec:
        model = import_class(spec)(**params)
    else:
        raise ValueError(
            "checkpoint does not record a model class; pass model_cls=...")
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, config
