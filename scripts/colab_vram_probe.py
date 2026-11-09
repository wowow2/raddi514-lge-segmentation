"""Measure ScarNet VRAM on a CUDA VM with synthetic slices."""

from __future__ import annotations

import os
import sys
import tarfile
import time
from pathlib import Path

import torch
import torch.nn as nn


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return 1

    pkg = os.environ.get("SCARNET_PKG", "/content/scarnet_pkg.tar.gz")
    dest = os.environ.get("SCARNET_DEST", "/content/scarnet_pkg")
    if Path(pkg).is_file():
        Path(dest).mkdir(parents=True, exist_ok=True)
        with tarfile.open(pkg) as archive:
            archive.extractall(dest)
    sys.path.insert(0, dest)

    try:
        from scarnet.model import ScarNet
    except ImportError as exc:
        print(f"cannot import scarnet: {exc}")
        print("install segment-anything: pip install "
              "git+https://github.com/facebookresearch/segment-anything.git")
        return 1

    batch_sizes = [int(x) for x in os.environ.get(
        "BATCH_SIZES", "1,2,4").split(",") if x.strip()]
    out_size = int(os.environ.get("OUT_SIZE", "128"))
    sam_input_size = int(os.environ.get("SAM_INPUT_SIZE", "1024"))
    unfreeze_last_n = int(os.environ.get("UNFREEZE_LAST_N", "0"))
    grad_checkpoint = _env_bool("GRAD_CHECKPOINT", False)
    amp = _env_bool("AMP", True)
    steps = int(os.environ.get("STEPS", "2"))
    unet_features = int(os.environ.get("UNET_FEATURES", "64"))

    props = torch.cuda.get_device_properties(0)
    print(f"gpu={props.name} total={props.total_memory / 1e9:.1f}GB "
          f"torch={torch.__version__}")
    print(f"out_size={out_size} sam_input_size={sam_input_size} "
          f"unfreeze_last_n={unfreeze_last_n} grad_checkpoint={grad_checkpoint} "
          f"amp={amp} steps={steps}")

    model = ScarNet(
        num_classes=5,
        medsam_ckpt=None,
        unfreeze_last_n=unfreeze_last_n,
        sam_input_size=sam_input_size,
        grad_checkpoint=grad_checkpoint,
        unet_features=unet_features,
    ).cuda()
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss()

    for batch_size in batch_sizes:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            for _ in range(steps):
                image = torch.rand(
                    batch_size, 1, out_size, out_size, device="cuda")
                mask = torch.randint(
                    0, 5, (batch_size, out_size, out_size), device="cuda")
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(image)
                    loss = criterion(logits, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = (time.time() - started) / max(steps, 1)
            allocated = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            print(f"batch={batch_size} ok step={elapsed:.2f}s "
                  f"peak_alloc={allocated:.2f}GB peak_reserved={reserved:.2f}GB")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"batch={batch_size} OOM")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
