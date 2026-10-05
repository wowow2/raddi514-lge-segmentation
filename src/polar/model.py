import torch
import torch.nn as nn
import torch.nn.functional as F

class CircularConv2d(nn.Module):
    """
    Circular Convolution that handles circular padding of theta and replicate padding of r
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int | tuple[int, int] = 1,):
        super().__init__()
        self.pad_h = self.pad_w = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Theta padded circularly, radius padded with replicate
        x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="replicate")
        return self.conv(x)

class ConvBlock(nn.Module):
    """
    nnUNet style ConvBlock with sequential circular convolutions, instance norm, and leakyrelu
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int | tuple[int, int] = 1):
        super().__init__()

        self.block = nn.Sequential(
            CircularConv2d(in_channels, out_channels, kernel_size=3, stride=stride),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            CircularConv2d(out_channels, out_channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class SeamUpsample(nn.Module):
    """Upsample (B, C, R, Θ) with circular wrap along Θ, then project channels with a 1x1 conv."""
    def __init__(self, in_channels: int, out_channels: int, scale_factor: tuple[int, int] = (2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.st = scale_factor[1]
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1,1,0,0), mode="circular")
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        x = x[:, :, :, self.st : -self.st]
        return self.conv1x1(x)

class PolarUNet(nn.Module):
    """
    nnU-Net-style 2D U-Net for polar LV slices (B, C, R, Θ) -> list of 5-class logits, with circular padding along Θ.
    Polar U-Net: (B, C, 64, 256) -> [final logits, *aux logits in train mode], circular in Θ.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        use_radius: bool = False,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.use_radius = use_radius
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # If use_radius is True, we concatenate an extra coordinate channel to the input
        stem_in = in_channels + 1 if use_radius else in_channels

        # Encoder
        self.enc0 = ConvBlock(stem_in, 32, stride=1)
        self.enc1 = ConvBlock(32, 64, stride=(1, 2))
        self.enc2 = ConvBlock(64, 128, stride=(2, 2))
        self.enc3 = ConvBlock(128, 256, stride=(2, 2))
        self.enc4 = ConvBlock(256, 256, stride=(1, 2))  # Bottleneck

        # Upsamples
        self.up3 = SeamUpsample(256, 256, scale_factor=(1, 2))
        self.up2 = SeamUpsample(256, 128, scale_factor=(2, 2))
        self.up1 = SeamUpsample(128, 64, scale_factor=(2, 2))
        self.up0 = SeamUpsample(64, 32, scale_factor=(1, 2))

        # Decoder
        self.dec3 = ConvBlock(256 + 256, 256, stride=1)
        self.dec2 = ConvBlock(128 + 128, 128, stride=1)
        self.dec1 = ConvBlock(64 + 64, 64, stride=1)
        self.dec0 = ConvBlock(32 + 32, 32, stride=1)

        # Output Heads
        self.head_final = nn.Conv2d(32, num_classes, kernel_size=1)
        if self.deep_supervision:
            self.head_aux1 = nn.Conv2d(64, num_classes, kernel_size=1)
            self.head_aux2 = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        assert x.shape[-2] % 4 == 0 and x.shape[-1] % 16 == 0, (
            f"Input spatial shape {x.shape[-2:]} must be divisible by (4, 16)"
        )

        if self.use_radius:
            B, _, R, T = x.shape
            r = torch.linspace(0.0, 1.0, R, device=x.device, dtype=x.dtype)
            r = r.view(1, 1, R, 1).expand(B, 1, R, T)
            x = torch.cat([x, r], dim=1)

        e0 = self.enc0(x)   # (B, 32,  64, 256)
        e1 = self.enc1(e0)  # (B, 64,  64, 128)
        e2 = self.enc2(e1)  # (B, 128, 32,  64)
        e3 = self.enc3(e2)  # (B, 256, 16,  32)
        e4 = self.enc4(e3)  # (B, 256, 16,  16) (Bottleneck)

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))  # (B, 256, 16, 32)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 128, 32, 64)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 64,  64, 128)
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))  # (B, 32,  64, 256)

        out = [self.head_final(d0)]
        if self.deep_supervision and self.training:
            out += [self.head_aux1(d1), self.head_aux2(d2)]
        return out



