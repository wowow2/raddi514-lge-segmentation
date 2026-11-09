"""Model-agnostic 2D trainer for the ScarNet stage-2 refiner."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import (
    DEFAULT_GT_ROOT,
    DEFAULT_SCARNET_MODEL_DIR,
    DEFAULT_SCARNET_PRED_DIR,
    DEFAULT_SPLITS_PATH,
    repo_path,
)
from src.datautils.splits import CVSplits
from src.layout import artifact_id
from src.runs import RunManifest, provenance_now, read_manifest, write_manifest

from .checkpoint import import_class, load_checkpoint
from .dataset import ScarNetDataset
from .loss import CombinedLoss


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


class ScarNetTrainer:
    """Train/validate/checkpoint one fold of an injected 2D segmentation model."""

    def __init__(
        self,
        fold: int = 0,
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 1e-3,
        lr_encoder: float | None = None,
        weight_decay: float = 1e-4,
        device: str | None = None,
        save_dir: str = DEFAULT_SCARNET_MODEL_DIR,
        pred_dir: str | None = DEFAULT_SCARNET_PRED_DIR,
        splits_path: str = DEFAULT_SPLITS_PATH,
        gt_root: str = DEFAULT_GT_ROOT,
        num_workers: int = 2,
        seed: int = 42,
        resume: bool = False,
        model: torch.nn.Module | None = None,
        model_class: str | None = None,
        model_params: dict | None = None,
        criterion: torch.nn.Module | None = None,
        train_ds=None,
        val_ds=None,
        out_size: int = 128,
        pad: int = 16,
        min_size: int = 64,
        multiple: int = 16,
        clip_pct: tuple[float, float] = (1.0, 99.0),
        use_largest_component: bool = True,
        amp: bool = True,
        grad_accum: int = 1,
        select_metric: str = "lesion",
        augment: bool = True,
        run_id: str | None = None,
        tags: tuple[str, ...] | list[str] = (),
        description: str | None = None,
        write_metadata: bool = True,
        model_name: str | None = None,
    ):
        if model is None and model_class is None:
            raise ValueError(
                "Pass model=... or model_class='pkg.module:ClassName ...'; "
                "no network architecture is bundled with the ScarNet plumbing.")
        if grad_accum < 1:
            raise ValueError("grad_accum must be >= 1")
        if select_metric not in ("lesion", "mean", "loss", "scar"):
            raise ValueError(
                "select_metric must be one of lesion/mean/loss/scar, "
                f"got {select_metric!r}")

        self.fold = fold
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_encoder = lr_encoder
        self.weight_decay = weight_decay
        self.seed = seed
        self.resume = resume
        self.write_metadata = write_metadata
        self.out_size = out_size
        self.pad = pad
        self.min_size = min_size
        self.multiple = multiple
        self.clip_pct = tuple(clip_pct)
        self.use_largest_component = bool(use_largest_component)
        self.grad_accum = grad_accum
        self.select_metric = select_metric
        self.augment = augment

        self.model_params = dict(model_params or {})
        if model is not None:
            self.model_class = model_class or f"{type(model).__module__}:{type(model).__name__}"
        else:
            self.model_class = model_class
        self.model_name = model_name or (
            self.model_class.split(":")[-1] if self.model_class else "scarnet")

        set_seed(seed)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.save_dir = Path(repo_path(save_dir))
        self.fold_dir = self.save_dir / f"fold_{fold}"
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path = self.fold_dir / "checkpoint_best.pt"
        self.last_ckpt_path = self.fold_dir / "checkpoint_last.pt"

        pred_dir_resolved = str(repo_path(pred_dir)) if pred_dir else None
        splits_path = str(repo_path(splits_path))
        gt_root = str(repo_path(gt_root))

        self.train_ds = train_ds if train_ds is not None else ScarNetDataset(
            gt_root=gt_root, pred_dir=pred_dir_resolved,
            splits_path=splits_path, fold=fold, split="train",
            pad=pad, out_size=out_size, min_size=min_size, multiple=multiple,
            clip_pct=clip_pct, use_largest_component=use_largest_component)
        self.val_ds = val_ds if val_ds is not None else ScarNetDataset(
            gt_root=gt_root, pred_dir=pred_dir_resolved,
            splits_path=splits_path, fold=fold, split="val",
            pad=pad, out_size=out_size, min_size=min_size, multiple=multiple,
            clip_pct=clip_pct, use_largest_component=use_largest_component)

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
            self.model = import_class(self.model_class)(
                **self.model_params).to(self.device)

        num_classes = int(self.model_params.get(
            "num_classes", getattr(self.model, "num_classes", 5)))
        self.num_classes = num_classes
        self.criterion = (criterion or CombinedLoss(num_classes=num_classes)).to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(epochs, 1), eta_min=1e-6)

        self.use_amp = bool(amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.best_score = float("-inf")
        self.start_epoch = 0
        self.history: dict[str, list] = {
            "train_loss": [], "val_loss": [], "lr": [], "val_score": [],
            "val_dice": [],
        }

        self.config = self._build_config(
            gt_root=gt_root, pred_dir=pred_dir_resolved,
            splits_path=splits_path, run_id=run_id)

        if self.resume:
            self._load_last()
        if self.write_metadata:
            self._write_metadata(pred_dir=pred_dir_resolved, run_id=run_id,
                                 tags=tags, description=description)
        with open(self.fold_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

    def _build_optimizer(self):
        """Use the model's differential-LR groups when it exposes them."""
        if self.lr_encoder is None:
            return optim.Adam(self.model.parameters(), lr=self.lr,
                              weight_decay=self.weight_decay)
        builder = getattr(self.model, "param_groups", None)
        if callable(builder):
            try:
                groups = builder(lr_encoder=self.lr_encoder, lr_new=self.lr,
                                 weight_decay=self.weight_decay)
                return optim.Adam(groups)
            except TypeError as exc:
                raise TypeError(
                    f"model.param_groups() does not accept lr_encoder/lr_new/"
                    f"weight_decay: {exc}") from exc
        print("warning: --lr-encoder was set but the model has no param_groups(); "
              "using one LR for all parameters")
        return optim.Adam(self.model.parameters(), lr=self.lr,
                          weight_decay=self.weight_decay)

    # ------------------------------------------------------------- metadata
    def _build_config(self, gt_root: str, pred_dir: str | None,
                      splits_path: str, run_id: str | None) -> dict:
        splits = None
        try:
            splits = CVSplits.load(splits_path, validate=False)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return {
            "run_id": run_id or artifact_id(self.save_dir),
            "model": {
                "name": self.model_name,
                "class": self.model_class,
                "params": self.model_params,
            },
            "data": {
                "gt_root": gt_root,
                "pred_dir": pred_dir,
                "splits_path": splits_path,
                "splits_sha256": splits.sha256 if splits else None,
                "n_folds": len(splits.folds) if splits else None,
                "fold": self.fold,
                "out_size": self.out_size,
                "pad": self.pad,
                "min_size": self.min_size,
                "multiple": self.multiple,
                "clip_pct": list(self.clip_pct),
                "use_largest_component": self.use_largest_component,
                "train_slices": len(self.train_ds),
                "val_slices": len(self.val_ds),
            },
            "training": {
                "seed": self.seed,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "lr_encoder": self.lr_encoder,
                "weight_decay": self.weight_decay,
                "optimizer": "Adam",
                "scheduler": "CosineAnnealingLR",
                "loss": type(self.criterion).__name__,
                "amp": self.use_amp,
                "grad_accum": self.grad_accum,
                "augment": "flips_rot90" if self.augment else None,
                "select_metric": self.select_metric,
            },
        }

    def _write_metadata(self, pred_dir: str | None, run_id: str | None,
                        tags, description: str | None) -> None:
        path = self.save_dir / "run.json"
        manifest = RunManifest.from_dict(read_manifest(path))
        manifest.run_id = run_id or manifest.run_id or artifact_id(self.save_dir)
        manifest.stage = manifest.stage or "stage2"
        manifest.description = description or manifest.description
        manifest.tags = sorted(set(manifest.tags) | set(tags))
        if pred_dir:
            manifest.source_run = manifest.source_run or artifact_id(pred_dir)
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
        manifest.provenance = provenance_now("scripts/train_scarnet.py")
        write_manifest(path, manifest)

    # ------------------------------------------------------------ checkpoint
    def _checkpoint_payload(self, epoch: int) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "best_score": self.best_score,
            "history": self.history,
            "config": self.config,
        }

    def _load_last(self) -> None:
        if not self.last_ckpt_path.exists():
            print(f"resume requested but {self.last_ckpt_path} is absent; starting fresh")
            return
        ckpt = load_checkpoint(self.last_ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.best_score = float(ckpt.get("best_score", float("-inf")))
        self.history = ckpt.get("history", self.history)
        print(f"resumed from epoch {self.start_epoch} ({self.last_ckpt_path})")

    # -------------------------------------------------------------- training
    def _autocast(self):
        return torch.amp.autocast("cuda", enabled=self.use_amp)

    @staticmethod
    def _logits(outputs):
        return outputs[0] if isinstance(outputs, (list, tuple)) else outputs

    @staticmethod
    def _augment(img: torch.Tensor, mask: torch.Tensor):
        if random.random() < 0.5:
            img = torch.flip(img, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        if random.random() < 0.5:
            img = torch.flip(img, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))
        k = random.randint(0, 3)
        if k:
            img = torch.rot90(img, k, dims=(-2, -1))
            mask = torch.rot90(mask, k, dims=(-2, -1))
        return img, mask

    def train_epoch(self) -> float:
        self.model.train()
        total, seen = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            img = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            if self.augment:
                img, mask = self._augment(img, mask)

            with self._autocast():
                logits = self._logits(self.model(img))
                loss = self.criterion(logits, mask)

            self.scaler.scale(loss / self.grad_accum).backward()
            if (step + 1) % self.grad_accum == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total += loss.item() * img.size(0)
            seen += img.size(0)

        if len(self.train_loader) % self.grad_accum:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return total / max(seen, 1)

    @torch.no_grad()
    def validate_epoch(self) -> tuple[float, dict[int, float]]:
        self.model.eval()
        total, seen = 0.0, 0
        classes = max(self.num_classes, 1)
        inter = torch.zeros(classes)
        psum = torch.zeros(classes)
        tsum = torch.zeros(classes)
        for batch in self.val_loader:
            img = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            logits = self._logits(self.model(img))
            total += self.criterion(logits, mask).item() * img.size(0)
            seen += img.size(0)

            pred = logits.argmax(dim=1)
            for c in range(1, classes):
                p, t = pred == c, mask == c
                inter[c] += (p & t).sum().item()
                psum[c] += p.sum().item()
                tsum[c] += t.sum().item()
        dice = {
            c: float((2 * inter[c] + 1e-5) / (psum[c] + tsum[c] + 1e-5))
            for c in range(1, classes)
        }
        return total / max(seen, 1), dice

    def _score(self, val_loss: float, dice: dict[int, float]) -> float:
        if self.select_metric == "loss":
            return -val_loss
        if self.select_metric == "scar":
            return dice.get(3, 0.0)
        if self.select_metric == "mean":
            return float(np.mean(list(dice.values()))) if dice else 0.0
        return 0.5 * (dice.get(3, 0.0) + dice.get(4, 0.0))

    def fit(self) -> Path:
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_loss, dice = self.validate_epoch()
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            score = self._score(val_loss, dice)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)
            self.history["val_score"].append(score)
            self.history["val_dice"].append(dice)

            if score > self.best_score:
                self.best_score = score
                _atomic_torch_save(
                    self._checkpoint_payload(epoch), self.best_ckpt_path)
            _atomic_torch_save(self._checkpoint_payload(epoch), self.last_ckpt_path)

            dice_text = " ".join(
                f"{name} {dice.get(idx, float('nan')):.3f}"
                for idx, name in ((1, "cav"), (2, "myo"), (3, "scar"), (4, "mvo")))
            print(f"ep {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                  f"{dice_text} | score {score:.4f} | lr {lr:.1e} | "
                  f"{time.time() - t0:.0f}s")

        with open(self.fold_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        return self.best_ckpt_path
