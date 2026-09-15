"""Losses for RAW restoration.

Plain L2 is a poor fit here. RAW data spans four orders of magnitude of
brightness, so an L2 loss is dominated by the highlights and the shadows -- the
part people actually judge a denoiser on -- barely contribute to the gradient.
The mitigations used by the literature, all available here:

* **Charbonnier** (Lai et al., CVPR 2017): a differentiable L1 that is far less
  highlight-dominated than L2 and does not blur edges the way L2 does.
* **PSNR loss** (NAFNet): optimise the evaluation metric directly.
* **Frequency loss**: an L1 penalty on the FFT magnitude, which catches the
  periodic banding residue that a pixel-space loss is nearly blind to.
"""

from __future__ import annotations

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


class PSNRLoss(nn.Module):
    """Negative PSNR, as used to train NAFNet on SIDD."""

    def __init__(self, max_val: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.scale = 10.0 / torch.log(torch.tensor(10.0))
        self.max_val = max_val
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = ((pred - target) ** 2).flatten(1).mean(dim=1) + self.eps
        return -(self.scale.to(pred.device) * torch.log(self.max_val ** 2 / mse)).mean()


class FrequencyLoss(nn.Module):
    """L1 between the FFT magnitudes of prediction and target."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = torch.fft.rfft2(pred.float(), norm="ortho")
        ft = torch.fft.rfft2(target.float(), norm="ortho")
        return (torch.view_as_real(fp) - torch.view_as_real(ft)).abs().mean()


class ShadowWeightedLoss(nn.Module):
    """Charbonnier reweighted so dark pixels are not drowned out.

    Weighting by ``1 / (target + offset)`` approximates a perceptual (gamma-like)
    space without leaving the linear domain the network operates in. Deep shadows
    are where high-ISO RAW denoising is won or lost.
    """

    def __init__(self, offset: float = 0.05, eps: float = 1e-3):
        super().__init__()
        self.offset = offset
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = 1.0 / (target.detach().clamp(min=0.0) + self.offset)
        w = w / w.mean()
        return (w * torch.sqrt((pred - target) ** 2 + self.eps2)).mean()


class CombinedLoss(nn.Module):
    """Weighted sum of the above. Weights of zero skip the term entirely."""

    def __init__(self, charbonnier: float = 1.0, psnr: float = 0.0,
                 frequency: float = 0.0, shadow: float = 0.0):
        super().__init__()
        self.w = {"charbonnier": charbonnier, "psnr": psnr,
                  "frequency": frequency, "shadow": shadow}
        self.charbonnier = CharbonnierLoss()
        self.psnr = PSNRLoss()
        self.frequency = FrequencyLoss()
        self.shadow = ShadowWeightedLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        total = pred.new_zeros(())
        parts = {}
        for name, weight in self.w.items():
            if weight == 0:
                continue
            value = getattr(self, name)(pred, target)
            parts[name] = float(value.detach())
            total = total + weight * value
        return total, parts


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1).clamp(min=1e-12)
    return (10.0 * torch.log10(max_val ** 2 / mse)).mean()
