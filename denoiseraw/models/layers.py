"""Shared building blocks for the restoration networks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over ``(B, C, H, W)``.

    ``nn.LayerNorm`` would need a permute round-trip; doing it directly keeps the
    memory layout and is measurably faster in the inner loop.
    """

    def __init__(self, channels: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            # BiasFree variant (Restormer): subtracting the mean removes the
            # signal's DC level, which for restoration throws away information
            # the network needs to stay linear in the input.
            sigma = x.var(dim=1, keepdim=True, unbiased=False)
            out = x / torch.sqrt(sigma + self.eps)
            return out * self.weight.view(1, -1, 1, 1)
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.var(dim=1, keepdim=True, unbiased=False)
        out = (x - mu) / torch.sqrt(sigma + self.eps)
        return out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """NAFNet's activation: split the channels in half and multiply.

    The point of the paper (Chen et al., *Simple Baselines for Image
    Restoration*, ECCV 2022) is that this multiplicative gate replaces GELU
    entirely -- nonlinearity comes from the product, not from an activation
    function -- and it is cheaper.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class SimplifiedChannelAttention(nn.Module):
    """Squeeze-and-excitation with the nonlinearity removed (NAFNet's SCA)."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class PixelShuffleUp(nn.Module):
    """2x upsample via channel expansion + pixel shuffle (no checkerboard)."""

    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels, out_channels * 4, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def pad_for_divisibility(x: torch.Tensor, factor: int):
    """Reflect-pad so a U-Net of ``log2(factor)`` levels divides evenly."""
    _, _, h, w = x.shape
    ph, pw = (-h) % factor, (-w) % factor
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)


def unpad(x: torch.Tensor, size) -> torch.Tensor:
    h, w = size
    return x[..., :h, :w]
