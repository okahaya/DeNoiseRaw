"""Noise-conditioned wrapper around any backbone.

A denoiser that sees only noisy pixels has to *infer* the noise level from
texture. That inference is what makes blind models over-smooth clean images and
under-clean grainy ones. Telling the network the noise level instead (FFDNet,
Zhang et al. 2018; CBDNet, Guo et al. 2019) removes the guess -- and for RAW we
can do better than a scalar, because the calibrated Poisson-Gaussian law gives a
*per-pixel* standard deviation. The network is handed exactly how uncertain each
pixel is.

Conditioning modes
------------------
``sigma_map``
    Concatenate ``sqrt(a*x + b)`` as extra input planes. The signal path stays
    linear, which keeps highlight roll-off physically correct. Default.
``vst``
    Generalised Anscombe transform in, closed-form unbiased inverse out. The
    network then only ever sees unit-variance noise, so a single model covers
    every ISO. The inverse amplifies error in deep shadows, which is why it is
    not the default.
``none``
    Blind. Only useful for ablations and for self-supervised fine-tuning where
    no profile is available.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..vst import gat, gat_inverse

CONDITIONING_MODES = ("sigma_map", "vst", "none")


def extra_input_channels(conditioning: str, channels: int) -> int:
    """How many planes the conditioning adds to the backbone's input."""
    return channels if conditioning == "sigma_map" else 0


class DenoiserWrapper(nn.Module):
    """Adds noise conditioning and size padding to a restoration backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        conditioning: str = "sigma_map",
        channels: int = 4,
        sigma_gain: float = 20.0,
    ):
        super().__init__()
        if conditioning not in CONDITIONING_MODES:
            raise ValueError(f"unknown conditioning {conditioning!r}; expected one of {CONDITIONING_MODES}")
        self.backbone = backbone
        self.conditioning = conditioning
        self.channels = channels
        # Typical sigma is 2e-3..3e-2 while the signal spans 0..1. Scaling the
        # sigma planes into a comparable range keeps the first convolution's
        # gradients balanced across its input channels.
        self.sigma_gain = sigma_gain

    @property
    def padder_size(self) -> int:
        return getattr(self.backbone, "padder_size", 16)

    def sigma_map(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(a * torch.clamp(x, min=0.0) + b, min=1e-12))

    def forward(
        self,
        x: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise ``x``.

        Parameters
        ----------
        x:
            ``(B, C, H, W)`` packed RAW planes in normalised units.
        a, b:
            Poisson-Gaussian parameters broadcastable against ``x`` -- usually
            ``(B, C, 1, 1)``. Required unless ``conditioning='none'``.
        """
        _, _, h, w = x.shape
        pad = self.padder_size
        ph, pw = (-h) % pad, (-w) % pad
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        if self.conditioning == "none":
            out = self.backbone(x)

        elif self.conditioning == "sigma_map":
            if a is None or b is None:
                raise ValueError("sigma_map conditioning requires the noise parameters a and b")
            sigma = self.sigma_map(x, a, b)
            out = self.backbone(torch.cat([x, sigma * self.sigma_gain], dim=1))

        else:  # vst
            if a is None or b is None:
                raise ValueError("vst conditioning requires the noise parameters a and b")
            a = torch.clamp(a, min=1e-9)
            out = gat_inverse(self.backbone(gat(x, a, b)), a, b)

        return out[..., :h, :w]
