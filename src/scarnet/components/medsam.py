"""ScarNet MedSAM branch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import ConvGNReLU, SELayer, ScarAttention

MEDSAM_NECK_CHANNELS = 256
MEDSAM_FEATURE_CHANNELS = (128, 64, 32)


def build_sam_vit_b_encoder() -> nn.Module:
    """Build a bare SAM ViT-B ``image_encoder`` (random weights)."""
    from segment_anything import sam_model_registry

    return sam_model_registry["vit_b"](checkpoint=None).image_encoder


def load_medsam_encoder_weights(encoder: nn.Module, ckpt_path: str) -> None:
    """Load ``image_encoder.*`` weights from a MedSAM/SAM checkpoint."""
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict):
        for key in ("model", "state_dict"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break

    prefixes = ("module.image_encoder.", "image_encoder.")
    enc_state = {}
    prefixed = False
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                enc_state[key[len(prefix):]] = value
                prefixed = True
                break
    if not prefixed:
        enc_state = state
    if not any(k.startswith("patch_embed.") for k in enc_state):
        raise RuntimeError(
            f"{ckpt_path} does not contain SAM image-encoder weights")

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"MedSAM checkpoint {ckpt_path} does not match the ViT-B image encoder: "
            f"{len(missing)} missing keys (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")


class MedSAMBranch(nn.Module):
    """ViT-B encoder plus the conv/SE/ScarAttention feature decoder."""

    def __init__(
        self,
        ckpt_path: str | None = None,
        unfreeze_last_n: int = 0,
        sam_input_size: int = 1024,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        encoder = build_sam_vit_b_encoder()
        if ckpt_path:
            load_medsam_encoder_weights(encoder, ckpt_path)
        self.image_encoder = encoder
        self.sam_input_size = sam_input_size
        self.grad_checkpoint = grad_checkpoint

        depth = len(encoder.blocks)
        if not 0 <= unfreeze_last_n <= depth:
            raise ValueError(
                f"unfreeze_last_n must be in [0, {depth}], got {unfreeze_last_n}")
        self.n_frozen = depth - unfreeze_last_n

        for param in encoder.parameters():
            param.requires_grad = False
        for block in encoder.blocks[self.n_frozen:]:
            for param in block.parameters():
                param.requires_grad = True
        if unfreeze_last_n > 0:
            for param in encoder.neck.parameters():
                param.requires_grad = True

        def decoder_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                ConvGNReLU(in_ch, out_ch),
                SELayer(out_ch),
                ScarAttention(out_ch),
            )

        self.decoder_blocks = nn.ModuleList()
        in_ch = MEDSAM_NECK_CHANNELS
        for out_ch in MEDSAM_FEATURE_CHANNELS:
            self.decoder_blocks.append(decoder_block(in_ch, out_ch))
            in_ch = out_ch

    def _pos_embed(self, tokens: torch.Tensor):
        pos = self.image_encoder.pos_embed
        if pos is None:
            return 0
        if pos.ndim == 3:
            side = int(pos.shape[1] ** 0.5)
            pos = pos.reshape(1, side, side, -1)
        if pos.shape[1:3] != tokens.shape[1:3]:
            pos = F.interpolate(
                pos.permute(0, 3, 1, 2),
                size=tokens.shape[1:3],
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        return pos

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) -> (B, 256, g, g) neck features."""
        encoder = self.image_encoder
        size = (self.sam_input_size, self.sam_input_size)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)

        all_frozen = self.n_frozen == len(encoder.blocks)
        with torch.no_grad():
            tokens = encoder.patch_embed(x)
            tokens = tokens + self._pos_embed(tokens)
            for block in encoder.blocks[: self.n_frozen]:
                tokens = block(tokens)
            if all_frozen:
                tokens = encoder.neck(tokens.permute(0, 3, 1, 2))
        if all_frozen:
            return tokens

        if not tokens.requires_grad:
            tokens = tokens.detach().requires_grad_(True)

        use_checkpoint = (
            self.grad_checkpoint and self.training and torch.is_grad_enabled())
        for block in encoder.blocks[self.n_frozen:]:
            tokens = (checkpoint(block, tokens, use_reentrant=False)
                      if use_checkpoint else block(tokens))
        return encoder.neck(tokens.permute(0, 3, 1, 2))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.encode(x)
        outputs = []
        for block in self.decoder_blocks:
            features = block(features)
            outputs.append(features)
        return outputs
