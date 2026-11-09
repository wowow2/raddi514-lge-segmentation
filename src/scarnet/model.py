"""ScarNet model assembly."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    MEDSAM_FEATURE_CHANNELS,
    AttentionUNet,
    ConvGNReLU,
    FusionBlock,
    MedSAMBranch,
    MultiScaleScarAttention,
)


class ScarNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        medsam_ckpt: str | None = None,
        unfreeze_last_n: int = 0,
        sam_input_size: int = 1024,
        grad_checkpoint: bool = False,
        unet_features: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.medsam = MedSAMBranch(
            ckpt_path=medsam_ckpt,
            unfreeze_last_n=unfreeze_last_n,
            sam_input_size=sam_input_size,
            grad_checkpoint=grad_checkpoint,
        )
        self.unet = AttentionUNet(num_classes, features=unet_features)

        reduced = MEDSAM_FEATURE_CHANNELS[-1]
        self.channel_reducer = nn.Conv2d(
            sum(MEDSAM_FEATURE_CHANNELS), reduced, 1)
        self.scar_attention = MultiScaleScarAttention(reduced)
        self.medsam_head = nn.Conv2d(reduced, num_classes, 1)
        self.fusion = FusionBlock(num_classes)

        self._init_new_layers()

    def _init_new_layers(self) -> None:
        for module in self.modules():
            if isinstance(module, ConvGNReLU):
                nn.init.kaiming_normal_(
                    module[0].weight, mode="fan_out", nonlinearity="relu")

    def encoder_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.medsam.image_encoder.parameters()
                if p.requires_grad]

    def new_parameters(self) -> list[nn.Parameter]:
        encoder_ids = {id(p) for p in self.medsam.image_encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in encoder_ids]

    def param_groups(self, lr_encoder: float = 1e-5, lr_new: float = 1e-3,
                     weight_decay: float = 0.0) -> list[dict]:
        groups = [{"params": self.new_parameters(), "lr": lr_new,
                   "weight_decay": weight_decay}]
        encoder = self.encoder_parameters()
        if encoder:
            groups.append({"params": encoder, "lr": lr_encoder,
                           "weight_decay": weight_decay})
        return groups

    def forward(self, x: torch.Tensor, return_branches: bool = False):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"expected input of shape (B, 1, H, W), got {tuple(x.shape)}")
        size = x.shape[-2:]

        features = [
            F.interpolate(f, size=size, mode="bilinear", align_corners=False)
            for f in self.medsam(x)
        ]
        fused_features = self.scar_attention(
            self.channel_reducer(torch.cat(features, dim=1)))
        sam_logits = self.medsam_head(fused_features)
        unet_logits = self.unet(x)

        logits = self.fusion(torch.cat([
            F.softmax(sam_logits, dim=1),
            F.softmax(unet_logits, dim=1),
        ], dim=1))
        if return_branches:
            return logits, sam_logits, unet_logits
        return logits
