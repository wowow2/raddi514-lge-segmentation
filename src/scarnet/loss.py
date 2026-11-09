"""ScarNet loss: Focal Tversky + Dice + weighted cross-entropy."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# EMIDEC order: background, cavity, myo, scar, MVO.
DEFAULT_CLASS_WEIGHTS = (0.1, 0.3, 0.3, 0.3, 0.3)


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss (Abraham & Khan, 2019)."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 2.0,
        smooth: float = 1e-6,
        include_background: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=1)
        num_classes = logits.shape[1]
        target_oh = F.one_hot(target, num_classes=num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)

        tp = (probs * target_oh).sum(dims)
        fp = (probs * (1.0 - target_oh)).sum(dims)
        fn = ((1.0 - probs) * target_oh).sum(dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = (1.0 - tversky) ** self.gamma

        present = target_oh.sum(dims) > 0
        if not self.include_background:
            present[0] = False
        loss = loss[present]
        return loss.mean() if loss.numel() else logits.sum() * 0.0


class DiceLoss(nn.Module):
    """Soft Dice loss over classes present in the target."""

    def __init__(self, smooth: float = 1e-6,
                 include_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=1)
        num_classes = logits.shape[1]
        target_oh = F.one_hot(target, num_classes=num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)

        intersection = (probs * target_oh).sum(dims)
        denom = probs.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice

        present = target_oh.sum(dims) > 0
        if not self.include_background:
            present[0] = False
        loss = loss[present]
        return loss.mean() if loss.numel() else logits.sum() * 0.0


class CombinedLoss(nn.Module):
    """Focal Tversky + Dice + weighted cross-entropy for 5-class EMIDEC.

    Args:
        lambda1 / lambda2 / lambda3: weights for Focal Tversky, Dice, CE.
        class_weights: CE class weights (background down-weighted).
        adaptive_scar: multiply lambda1 by ``scar_boost`` when the batch
            contains scar voxels, matching the public ScarNet trainer.
    """

    def __init__(
        self,
        lambda1: float = 0.5,
        lambda2: float = 0.4,
        lambda3: float = 0.1,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 2.0,
        smooth: float = 1e-6,
        class_weights=DEFAULT_CLASS_WEIGHTS,
        adaptive_scar: bool = True,
        scar_boost: float = 1.5,
        num_classes: int = 5,
        scar_label: int = 3,
        include_background: bool = False,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.adaptive_scar = adaptive_scar
        self.scar_boost = scar_boost
        self.scar_label = scar_label
        self.num_classes = num_classes

        self.focal_tversky = FocalTverskyLoss(
            alpha=alpha, beta=beta, gamma=gamma, smooth=smooth,
            include_background=include_background)
        self.dice = DiceLoss(
            smooth=smooth, include_background=include_background)

        weights = None
        if class_weights is not None:
            weights = torch.as_tensor(tuple(class_weights), dtype=torch.float32)
            if len(weights) != num_classes:
                raise ValueError(
                    f"class_weights has {len(weights)} entries, expected {num_classes}")
        # The criterion owns the buffer, so model.to(device) moves it with the
        # module; do not register the same tensor on the parent as well.
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(
        self,
        logits: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        target: torch.Tensor,
        return_components: bool = False,
    ):
        """Return the scalar loss, or ``(loss, components)`` when requested.

        A list/tuple of logits is treated as deep supervision: the first map is
        the full-resolution prediction and defines the reported components.
        """
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        ftl = self.focal_tversky(logits, target)
        dice = self.dice(logits, target)
        ce = self.ce(logits, target)

        lambda1 = self.lambda1
        if self.adaptive_scar and bool((target == self.scar_label).any()):
            lambda1 = lambda1 * self.scar_boost

        loss = lambda1 * ftl + self.lambda2 * dice + self.lambda3 * ce
        if not return_components:
            return loss
        return loss, {
            "loss": float(loss.detach()),
            "focal_tversky": float(ftl.detach()),
            "dice": float(dice.detach()),
            "cross_entropy": float(ce.detach()),
            "lambda1": lambda1,
        }

