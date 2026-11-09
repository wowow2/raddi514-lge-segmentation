"""ScarNet branch fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import ConvGNReLU


class FusionBlock(nn.Module):
    """3x3 convolution refinement followed by a 1x1 class projection."""

    def __init__(self, num_classes: int):
        super().__init__()
        channels = 2 * num_classes
        self.block = nn.Sequential(
            ConvGNReLU(channels, channels),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
