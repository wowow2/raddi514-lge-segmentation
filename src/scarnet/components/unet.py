"""ScarNet U-Net branch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ConvGNReLU, SELayer, ScarAttention


class UNetBlock(nn.Module):
    """Two conv/GroupNorm/ReLU layers, then SE and ScarAttention."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvGNReLU(in_ch, out_ch)
        self.conv2 = ConvGNReLU(out_ch, out_ch)
        self.se = SELayer(out_ch)
        self.scar_attention = ScarAttention(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return self.scar_attention(x)


class AttentionUNet(nn.Module):
    """4-level U-Net with SE + ScarAttention in every block."""

    def __init__(self, num_classes: int, features: int = 64):
        super().__init__()
        f = features
        self.enc1 = UNetBlock(1, f)
        self.enc2 = UNetBlock(f, f * 2)
        self.enc3 = UNetBlock(f * 2, f * 4)
        self.enc4 = UNetBlock(f * 4, f * 8)
        self.center = UNetBlock(f * 8, f * 16)
        self.dec4 = UNetBlock(f * 16 + f * 8, f * 8)
        self.dec3 = UNetBlock(f * 8 + f * 4, f * 4)
        self.dec2 = UNetBlock(f * 4 + f * 2, f * 2)
        self.dec1 = UNetBlock(f * 2 + f, f)
        self.final = nn.Conv2d(f, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self._up(c, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        return self.final(d1)
