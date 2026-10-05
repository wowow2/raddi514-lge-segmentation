import torch
import torch.nn as nn
import torch.nn.functional as F


class CompoundLoss(nn.Module):
    """CE + soft Dice over a list of logits (full-res first, then coarser aux heads)."""

    def __init__(
        self,
        num_classes: int = 5,
        weights: tuple[float, ...] = (1.0, 0.5, 0.25),
        eps: float = 1e-5,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weights = tuple(weights)
        self.eps = eps
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def _soft_targets(self, target: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """(B, R, Theta) int64 -> (B, C, *size) soft one-hot via area pooling."""
        oh = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        if oh.shape[-2:] == tuple(size):
            return oh
        return F.adaptive_avg_pool2d(oh, size)

    def _dice(self, logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        p = logits.softmax(1)
        dims = (0, 2, 3)  # Batch-level Dice per class across all spatial dimensions
        inter = (p * tgt).sum(dims)
        denom = p.sum(dims) + tgt.sum(dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice[1:].mean()  # Classes 1-4 only (Background excluded)

    def forward(self, outputs: list[torch.Tensor] | torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(outputs):
            outputs = [outputs]

        w = self.weights[:len(outputs)]
        total = 0.0

        for wi, logits in zip(w, outputs):
            tgt = self._soft_targets(target, logits.shape[-2:])
            ce = F.cross_entropy(logits, tgt)  # Soft-target CE
            dice = self._dice(logits, tgt)
            total = total + wi * (self.ce_weight * ce + self.dice_weight * dice)

        return total / sum(w)