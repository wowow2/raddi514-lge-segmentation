"""ScarNet attention components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def num_groups(channels: int, max_groups: int = 8) -> int:
    """Largest divisor of ``channels`` that is <= ``max_groups``."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with a channel-compatible group count."""
    return nn.GroupNorm(num_groups(channels), channels)


class ConvGNReLU(nn.Sequential):
    """3x3 conv -> GroupNorm -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            gn(out_ch),
            nn.ReLU(inplace=True),
        )


class SELayer(nn.Module):
    """Squeeze-and-Excitation channel recalibration."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c  = x.shape[:2]
        weights = self.fc(self.pool(x).flatten(1)).view(b,c,1,1)
        return x * weights


class ScarAttention(nn.Module):
    """Channel gate followed by a spatial gate."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 8, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_gate(self.pool(x))
        return x * self.spatial_gate(x)


class MultiScaleScarAttention(nn.Module):
    """ScarAttention at several pooling scales, fused by a 1x1 conv."""

    def __init__(self, channels: int, scales: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.scales = tuple(scales)
        self.attentions = nn.ModuleList(
            ScarAttention(channels) for _ in self.scales)
        self.fusion = nn.Conv2d(channels * len(self.scales), channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = []
        for scale, att in zip(self.scales, self.attentions):
            if scale > 1:
                y = F.adaptive_avg_pool2d(
                    x, (max(h // scale, 1), max(w // scale, 1)))
                y = att(y)
                y = F.interpolate(
                    y, size=(h, w), mode="bilinear", align_corners=False)
            else:
                y = att(x)
            outs.append(y)
        return self.fusion(torch.cat(outs, dim=1))
