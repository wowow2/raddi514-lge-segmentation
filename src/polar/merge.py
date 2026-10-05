import numpy as np
import torch
from scipy import ndimage as ndi

from src.datautils.postprocess import PostProcessor
from .dataset import PolarDataset
from .transforms import PolarTransform

CAV, MYO, SCAR, MVO = 1, 2, 3, 4
N_CLS = 5
_STRUCT = ndi.generate_binary_structure(2, 1)[:, :, None]  # in-plane only (10 mm slices)

def polar_probs_to_cartesian(
    polar_probs: dict[int, torch.Tensor],
    transform: PolarTransform,
    H: int, W: int, Z: int,
) -> np.ndarray:
    """{z: (5,R,Θ) softmax} -> (5,H,W,Z) float32. Zeros on slices the model skipped."""
    out = np.zeros((N_CLS, H, W, Z), np.float32)
    for z, p in polar_probs.items():
        cart = transform.to_cartesian(p, z=z, H=H, W=W)   # bilinear, (5,H,W)
        out[..., z] = cart.cpu().numpy()
    return out


def merge_polar_with_nnunet(
    nnunet_pred: np.ndarray,      # (H,W,Z) uint8 stage-1 labels
    cart_probs: np.ndarray,       # (5,H,W,Z) polar softmax in Cartesian
    covered_z: np.ndarray,        # (Z,) bool, slices the polar model ran on
    dilation: int = 2,
    disc_thresh: float = 0.5,
) -> np.ndarray:
    """Paint-over rule. Returns merged labels before the scar gate."""
    merged = nnunet_pred.copy()
    if not covered_z.any():
        return merged

    # voxels inside the disc
    in_disc = cart_probs.sum(0) >= disc_thresh

    # covered discs
    stage1_sm = np.isin(nnunet_pred, (SCAR, MVO)) & covered_z[None, None, :]
    merged[stage1_sm] = MYO

    # allowed region
    region = np.isin(nnunet_pred, (MYO, SCAR, MVO, CAV))
    if dilation > 0:
        region = ndi.binary_dilation(region, structure=_STRUCT, iterations=dilation)

    # polar scar/mvo paint
    polar_arg = cart_probs.argmax(0)
    paint = (np.isin(polar_arg, (SCAR, MVO)) & region & in_disc
             & covered_z[None, None, :])
    merged[paint] = polar_arg[paint].astype(np.uint8)
    return merged

class PolarMerger:
    """Run a trained PolarUNet on one case and merge into nnU-Net's labels."""

    def __init__(self, model, dilation: int = 2, gate: PostProcessor | None = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device).eval()
        self.dilation = dilation
        self.gate = gate if gate is not None else PostProcessor()
        self.device = device

    @torch.no_grad()
    def polar_probs(self, raw_vol, case_anchors, transform) -> dict[int, torch.Tensor]:
        """Softmax maps for every slice that has a predicted wall."""
        wall_z = case_anchors.anchors.myo_container.any(axis=(0, 1))
        zs = [z for z in range(raw_vol.shape[2]) if wall_z[z]]
        if not zs:
            return {}
        xs = []
        for z in zs:
            sl = torch.from_numpy(PolarDataset.normalize_slice(raw_vol[:, :, z])).float().unsqueeze(0)
            xs.append(transform.to_polar(sl, z=z))                  # (1,R,Θ)
        x = torch.stack(xs).to(self.device)                          # (N,1,R,Θ)
        probs = self.model(x)[0].float().softmax(1).cpu()            # (N,5,R,Θ)
        return {z: probs[i] for i, z in enumerate(zs)}

    def __call__(self, raw_vol, nnunet_pred, case_anchors, apply_gate: bool = True):
        H, W, Z = raw_vol.shape
        transform = PolarTransform(case_anchors.anchors)
        pp = self.polar_probs(raw_vol, case_anchors, transform)
        covered = np.zeros(Z, bool)
        covered[list(pp)] = True
        cart = polar_probs_to_cartesian(pp, transform, H, W, Z)
        merged = merge_polar_with_nnunet(nnunet_pred, cart, covered, self.dilation)
        return self.gate(merged) if apply_gate else merged




