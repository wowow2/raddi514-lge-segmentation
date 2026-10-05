"""Training orchestrator for Polar nnU-Net across cross-validation folds."""

import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import DEFAULT_GT_ROOT, DEFAULT_SPLITS_PATH
from .dataset import PolarDataset
from .loss import CompoundLoss
from .model import PolarUNet


class PolarTrainer:
    """Orchestrates training, validation, scheduling, and checkpointing for one fold."""

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
    ):
        self.fold = fold
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.use_radius = use_radius
        self.deep_supervision = deep_supervision

        # Device configuration
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Checkpoint directory
        self.save_dir = Path(save_dir) / f"fold_{fold}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path = self.save_dir / "checkpoint_best.pt"

        # DataLoaders
        train_ds = PolarDataset(
            gt_root=gt_root,
            pred_dir=pred_dir,
            splits_path=splits_path,
            fold=fold,
            split="train",
        )
        val_ds = PolarDataset(
            gt_root=gt_root,
            pred_dir=pred_dir,
            splits_path=splits_path,
            fold=fold,
            split="val",
        )

        pin_memory = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        # Model & Loss Criterion
        self.model = PolarUNet(
            in_channels=1,
            num_classes=5,
            use_radius=use_radius,
            deep_supervision=deep_supervision,
        ).to(self.device)

        self.criterion = CompoundLoss(num_classes=5)

        # Optimizer & Cosine LR Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )

        # State Tracking
        self.best_val_loss = float("inf")
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }
        self.best_score = -1.0

    def _roll_aug(self, img: torch.Tensor, tgt: torch.Tensor):
        """Same random Θ-roll for image and target, per sample (0..Θ-1, not just multiples of 16)."""
        T = img.shape[-1]
        for i, k in enumerate(torch.randint(0, T, (img.size(0),)).tolist()):
            img[i] = torch.roll(img[i], k, dims=-1)
            tgt[i] = torch.roll(tgt[i], k, dims=-1)
        return img, tgt

    def train_epoch(self) -> float:
        """Run one training epoch over self.train_loader. Returns mean train loss."""
        self.model.train()
        total, n = 0.0, 0
        for batch in self.train_loader:
            img = batch["image"].to(self.device, non_blocking=True)  # (B,C,R,Θ)
            tgt = batch["target"].to(self.device, non_blocking=True)  # (B,R,Θ) int64

            img, tgt = self._roll_aug(img, tgt)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(img)
            loss = self.criterion(outputs, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12.0)
            self.optimizer.step()
            total += loss.item() * img.size(0)
            n += img.size(0)
        return total / n

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
            outputs = self.model(img)  # eval: one-element list
            total += self.criterion(outputs, tgt).item() * img.size(0)
            n += img.size(0)

            pred = outputs[0].argmax(1)
            for c in range(1, 5):
                p, t = pred == c, tgt == c
                inter[c] += (p & t).sum().item()
                psum[c] += p.sum().item()
                tsum[c] += t.sum().item()
        dice = {c: float((2 * inter[c] + 1e-5) / (psum[c] + tsum[c] + 1e-5)) for c in range(1, 5)}
        return total / n, dice

    def fit(self) -> Path:
        for epoch in range(self.epochs):
            t0 = time.time()
            tr = self.train_epoch()
            va, dice = self.validate_epoch()
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            score = 0.5 * (dice[3] + dice[4])  # scar + MVO polar Dice
            self.history["train_loss"].append(tr)
            self.history["val_loss"].append(va)
            self.history["lr"].append(lr)
            self.history.setdefault("val_dice", []).append(dice)

            if score > self.best_score:
                self.best_score = score
                torch.save({"model": self.model.state_dict(), "epoch": epoch,
                            "score": score, "use_radius": self.use_radius,
                            "deep_supervision": self.deep_supervision},
                           self.best_ckpt_path)

            print(f"ep {epoch:03d} | train {tr:.4f} | val {va:.4f} | "
                  f"dice cav {dice[1]:.3f} myo {dice[2]:.3f} scar {dice[3]:.3f} mvo {dice[4]:.3f} | "
                  f"lr {lr:.1e} | {time.time() - t0:.0f}s")

        torch.save({"model": self.model.state_dict()}, self.save_dir / "checkpoint_last.pt")
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        return self.best_ckpt_path