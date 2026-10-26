"""Cartesian <-> Polar coordinate transformations using PyTorch grid_sample."""

import torch
import torch.nn.functional as F

from .anchors import AnchorData


class PolarTransform:
    def __init__(self, anchors: AnchorData, R: int = 64, Theta: int = 256):
        self.anchors = anchors
        self.R = R
        self.Theta = Theta

    def _resample(self, tensor: torch.Tensor, grid: torch.Tensor,
                out_spatial_shape: tuple[int, int], is_mask: bool = False, circular_pad_w: bool = False,) -> torch.Tensor:
        """
        Handles rank expansion, grid sample, and reshaping
        """
        orig_dtype = tensor.dtype

        # Expand to 4d (1, C, H, W) for grid sampling
        x = tensor.float()
        while x.ndim < 4:
            x = x.unsqueeze(0)

        if circular_pad_w:
            x = F.pad(x, (1, 1, 0, 0), mode="circular")

        mode= "nearest" if is_mask else "bilinear"
        sampled = F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=True)

        out_shape = (*tensor.shape[:-2], *out_spatial_shape)
        return sampled.to(orig_dtype).view(out_shape)

    def _build_forward_grid(self, z: int, H: int, W: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """
        Constructs polar sampling grid
        """
        xc, yc = self.anchors.centers[z]
        rmax = self.anchors.radii[z]

        # 1D physical axis
        r = torch.linspace(0.0, float(rmax), self.R, device=device)
        theta = (torch.arange(self.Theta, device=device)) / self.Theta * 2.0 * torch.pi - torch.pi

        # 2D coordinate matricies
        r_grid, theta_grid = torch.meshgrid(r, theta, indexing="ij")

        # Polar to Cartisian
        x = xc + r_grid * torch.cos(theta_grid)
        y = yc + r_grid * torch.sin(theta_grid)

        # Normalize of [-1.0, 1.0]
        norm_x = 2.0 * (x / (W-1)) - 1.0
        norm_y = 2.0 * (y / (H-1)) - 1.0

        return torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

    def _build_inverse_grid(
            self, z: int, H: int, W: int, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """
        Constructs Cartesian sampling grid
        """
        xc, yc = self.anchors.centers[z]
        rmax = self.anchors.radii[z]

        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")

        delta_x = x_grid - xc
        delta_y = y_grid - yc

        r = torch.hypot(delta_x, delta_y)
        theta = torch.atan2(delta_y, delta_x)

        norm_r = 2.0 * (r / rmax) - 1.0
        col = (theta + torch.pi) / (2.0 * torch.pi) * self.Theta  # [0, Theta]
        norm_theta = 2.0 * (col + 1.0) / (self.Theta + 1.0) - 1.0

        return torch.stack([norm_theta, norm_r], dim=-1).unsqueeze(0)

    def to_polar(
        self, slice_tensor: torch.Tensor, z: int, is_mask: bool = False
    ) -> torch.Tensor:
        """Warps a Cartesian slice -> unrolled Polar patch of shape (*, R, Theta)."""
        H, W = slice_tensor.shape[-2:]
        grid = self._build_forward_grid(z, H, W, device=slice_tensor.device)
        return self._resample(slice_tensor, grid, (self.R, self.Theta), is_mask)

    def to_cartesian(self, polar_tensor: torch.Tensor, z: int, H: int | None = None,
        W: int | None = None, is_mask: bool = False) -> torch.Tensor:
        """Warps an unrolled Polar patch -> Cartesian canvas of shape (*, H, W)."""
        if H is None or W is None:
            H, W = self.anchors.myo_container.shape[:2]

        grid = self._build_inverse_grid(z, H, W, device=polar_tensor.device)
        return self._resample(polar_tensor, grid, (H, W), is_mask, circular_pad_w=True)


