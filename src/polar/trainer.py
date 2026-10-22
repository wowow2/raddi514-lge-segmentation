"""Training orchestrator for the polar stage-2 model.

The trainer accepts an injected model, datasets, and loss so a new model can
be added without copying this file; the defaults reproduce the original
polar U-Net setup. Training settings and data identity are written to
``config.json`` and ``run.json`` and embedded in every checkpoint.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH
from src.datautils.splits import CVSplits
from src.runs import RunManifest, provenance_now
from .dataset import PolarDataset
from .loss import CompoundLoss
from .model import PolarUNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


class PolarTrainer:
    """Train/validate/checkpoint one fold of a (default: polar) 2D model."""

    def __init__(
        self,
        fold: int = 0,
        epochs: int = 60,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_radius: bool = False,
        deep_supervision: bool = True,
        device: str | None = None,
        save_dir: str = "var/polar_models",
        pred_dir: str = "var/nnUNet_validation_501_pp",
        splits_path: str = DEFAULT_SPLITS_PATH,
        gt_root: str = DEFAULT_GT_ROOT,
        num_workers: int = 2,
        seed: int = 42,
        resume: bool = False,
        model: torch.nn.Module | None = None,
        train_ds=None,
        val_ds=None,
        criterion: torch.nn.Module | None = None,
        run_id: str | None = None,
        tags: tuple[str, ...] | list[str] = (),
        description: str | None = None,
        write_metadata: bool = True,
        model_name: str = "polar_unet",
        model_params: dict | None = None,
    ):
        self.fold = fold
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_radius = use_radius
        self.deep_supervision = deep_supervision
        self.seed = seed
        self.resume = resume
        self.write_metadata = write_metadata
        self.model_name = model_name
        self.model_params = dict(model_params or {})

        set_seed(seed)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.save_dir = Path(save_dir)
        self.fold_dir = self.save_dir / f"fold_{fold}"
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path = self.fold_dir / "checkpoint_best.pt"
        self.last_ckpt_path = self.fold_dir / "checkpoint_last.pt"

        self.train_ds = train_ds if train_ds is not None else PolarDataset(
            gt_root=gt_root, pred_dir=pred_dir, splits_path=splits_path,
            fold=fold, split="train")
        self.val_ds = val_ds if val_ds is not None else PolarDataset(
            gt_root=gt_root, pred_dir=pred_dir, splits_path=splits_path,
            fold=fold, split="val")

        pin_memory = self.device.type == "cuda"
        self.train_loader = DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        self.val_loader = DataLoader(
            self.val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

        if model is not None:
            self.model = model.to(self.device)
        else:
            params = {
                "in_channels": 1, "num_classes": 5,
                "use_radius": use_radius,
                "deep_supervision": deep_supervision,
                **self.model_params,
            }
            self.model = PolarUNet(**params).to(self.device)
        self.criterion = criterion or CompoundLoss(num_classes=5)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6)

        self.best_val_loss = float("inf")
        self.best_score = -1.0
        self.start_epoch = 0
        self.history: dict[str, list] = {
            "train_loss": [], "val_loss": [], "lr": [], "val_dice": []}

        self.config = self._build_config(
            gt_root=gt_root, pred_dir=pred_dir, splits_path=splits_path,
            run_id=run_id)
        if self.resume:
            self._load_last()
        if self.write_metadata:
            self._write_metadata(pred_dir=pred_dir, run_id=run_id,
                                 tags=tags, description=description)

    # ------------------------------------------------------------- metadata
    def _build_config(self, gt_root: str, pred_dir: str, splits_path: str,
                      run_id: str | None) -> dict:
        splits = None
        try:
            splits = CVSplits.load(splits_path, validate=False)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        model_params = {
            "in_channels": 1, "num_classes": 5,
            "use_radius": self.use_radius,
            "deep_supervision": self.deep_supervision,
            **self.model_params,
        }
        return {
            "run_id": run_id or self.save_dir.resolve().name,
            "model": {"name": self.model_name, "params": model_params},
            "data": {
                "gt_root": str(gt_root),
                "pred_dir": str(pred_dir),
                "splits_path": str(splits_path),
                "splits_sha256": splits.sha256 if splits else None,
                "n_folds": len(splits.folds) if splits else None,
                "R": getattr(self.train_ds, "R", None),
                "Theta": getattr(self.train_ds, "Theta", None),
                "fold": self.fold,
                "train_slices": len(self.train_ds),
                "val_slices": len(self.val_ds),
            },
            "training": {
                "seed": self.seed,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "loss": {"name": type(self.criterion).__name__},
                "augment": "theta_roll",
            },
        }

    def _write_metadata(self, pred_dir: str, run_id: str | None,
                        tags, description: str | None) -> None:
        path = self.save_dir / "run.json"
        existing = {}
        if path.exists():
            with open(path) as f:
                existing = json.load(f)
        manifest = RunManifest.from_dict(existing)
        manifest.run_id = run_id or manifest.run_id or self.save_dir.resolve().name
        manifest.stage = manifest.stage or "stage2"
        manifest.description = description or manifest.description
        manifest.tags = sorted(set(manifest.tags) | set(tags))
        manifest.source_run = manifest.source_run or Path(pred_dir).name
        manifest.model = self.config["model"]
        fold_specific = {"fold", "train_slices", "val_slices"}
        manifest.data.update({k: v for k, v in self.config["data"].items()
                              if k not in fold_specific})
        folds = manifest.data.setdefault("folds", {})
        folds[str(self.fold)] = {
            "train_slices": self.config["data"]["train_slices"],
            "val_slices": self.config["data"]["val_slices"],
        }
        manifest.training.update(self.config["training"])
        manifest.paths["checkpoints"] = str(self.fold_dir.parent.resolve())
        manifest.provenance = provenance_now("scripts/train_polar.py")
        with open(self.fold_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        with open(path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def _checkpoint_payload(self, epoch: int) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_score": self.best_score,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config,
            # Legacy flags kept so older readers keep working.
            "use_radius": self.use_radius,
            "deep_supervision": self.deep_supervision,
        }

    def _load_last(self) -> None:
        if not self.last_ckpt_path.exists():
            print(f"resume requested but {self.last_ckpt_path} is absent; starting fresh")
            return
        try:
            ckpt = torch.load(str(self.last_ckpt_path), map_location=self.device,
                              weights_only=False)
        except TypeError:
            ckpt = torch.load(str(self.last_ckpt_path), map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.best_score = float(ckpt.get("best_score", -1.0))
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        self.history = ckpt.get("history", self.history)
        print(f"resumed from epoch {self.start_epoch} ({self.last_ckpt_path})")

    # -------------------------------------------------------------- training
    def _roll_aug(self, img: torch.Tensor, tgt: torch.Tensor):
        """Same random Θ-roll for image and target, per sample."""
        T = img.shape[-1]
        for i, k in enumerate(torch.randint(0, T, (img.size(0),)).tolist()):
            img[i] = torch.roll(img[i], k, dims=-1)
            tgt[i] = torch.roll(tgt[i], k, dims=-1)
        return img, tgt

    def train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        for batch in self.train_loader:
            img = batch["image"].to(self.device, non_blocking=True)
            tgt = batch["target"].to(self.device, non_blocking=True)
            img, tgt = self._roll_aug(img, tgt)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(img)
            loss = self.criterion(outputs, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12.0)
            self.optimizer.step()
            total += loss.item() * img.size(0)
            n += img.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def validate_epoch(self) -> tuple[float, dict[int, float]]:
        self.model.eval()
        total, n = 0.0, 0
        inter = torch.zeros(5)
        psum = torch.zeros(5)
        tsum = torch.zeros(5)
        for batch in self.val_loader:
            img = batch["image"].to(self.device, non_blocking=True)
            tgt = batch["target"].to(self.device, non_blocking=True)
            outputs = self.model(img)
            total += self.criterion(outputs, tgt).item() * img.size(0)
            n += img.size(0)

            pred = outputs[0].argmax(1)
            for c in range(1, 5):
                p, t = pred == c, tgt == c
                inter[c] += (p & t).sum().item()
                psum[c] += p.sum().item()
                tsum[c] += t.sum().item()
        dice = {c: float((2 * inter[c] + 1e-5) / (psum[c] + tsum[c] + 1e-5))
                for c in range(1, 5)}
        return total / max(n, 1), dice

    def fit(self) -> Path:
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            tr = self.train_epoch()
            va, dice = self.validate_epoch()
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            score = 0.5 * (dice[3] + dice[4])  # scar + MVO polar Dice
            self.history["train_loss"].append(tr)
            self.history["val_loss"].append(va)
            self.history["lr"].append(lr)
            self.history["val_dice"].append(dice)

            if score > self.best_score:
                self.best_score = score
                _atomic_torch_save(self._checkpoint_payload(epoch),
                                   self.best_ckpt_path)

            # Last is written every epoch so interrupted jobs can resume.
            _atomic_torch_save(self._checkpoint_payload(epoch),
                               self.last_ckpt_path)

            print(f"ep {epoch:03d} | train {tr:.4f} | val {va:.4f} | "
                  f"dice cav {dice[1]:.3f} myo {dice[2]:.3f} scar {dice[3]:.3f} mvo {dice[4]:.3f} | "
                  f"lr {lr:.1e} | {time.time() - t0:.0f}s")

        with open(self.fold_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        return self.best_ckpt_path
