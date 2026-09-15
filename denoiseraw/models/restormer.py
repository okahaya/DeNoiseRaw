"""Restormer -- *Efficient Transformer for High-Resolution Image Restoration*
(Zamir et al., CVPR 2022).

Self-attention costs O(HW^2) in the spatial domain, which is hopeless for a
24 MP RAW. Restormer's contribution is to apply attention across *channels*
instead: the attention map is C x C, so cost is linear in pixel count. Locality,
which channel attention alone would lose, comes from depth-wise 3x3 convolutions
inside both the attention and the feed-forward branch.

We keep this alongside NAFNet because its receptive field behaves differently on
the low-frequency chroma blotches that high-ISO DSLR files show -- the artefact
NAFNet is weakest on.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .layers import LayerNorm2d


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        # One learned temperature per head, replacing the fixed 1/sqrt(d) scale.
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        # L2-normalising q and k keeps the logits bounded; without it the
        # learned temperature makes training unstable at high resolution.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network."""

    def __init__(self, dim: int, expansion: float = 2.66, bias: bool = False):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, expansion: float = 2.66, bias: bool = False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim, bias=False)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim, bias=False)
        self.ffn = GDFN(dim, expansion, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class Downsample(nn.Module):
    """Halve resolution, double channels -- via PixelUnshuffle, so lossless."""

    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion: float = 2.66,
        bias: bool = False,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.out_channels = out_channels

        self.patch_embed = nn.Conv2d(in_channels, dim, 3, padding=1, bias=bias)

        self.encoder1 = nn.Sequential(*[TransformerBlock(dim, heads[0], ffn_expansion, bias)
                                        for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1], ffn_expansion, bias)
                                        for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(dim * 2)
        self.encoder3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2], ffn_expansion, bias)
                                        for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.Sequential(*[TransformerBlock(dim * 8, heads[3], ffn_expansion, bias)
                                      for _ in range(num_blocks[3])])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2], ffn_expansion, bias)
                                        for _ in range(num_blocks[2])])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1], ffn_expansion, bias)
                                        for _ in range(num_blocks[1])])
        self.up2_1 = Upsample(dim * 2)
        # Level 1 keeps 2*dim channels (no reduction) -- the paper's choice.
        self.decoder1 = nn.Sequential(*[TransformerBlock(dim * 2, heads[0], ffn_expansion, bias)
                                        for _ in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim * 2, heads[0], ffn_expansion, bias)
                                          for _ in range(num_refinement_blocks)])
        self.output = nn.Conv2d(dim * 2, out_channels, 3, padding=1, bias=bias)

        self.padder_size = 8

        if residual:
            # Start as the identity; see NAFNet's _zero_init for why.
            nn.init.zeros_(self.output.weight)
            if self.output.bias is not None:
                nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        e1 = self.encoder1(self.patch_embed(x))
        e2 = self.encoder2(self.down1_2(e1))
        e3 = self.encoder3(self.down2_3(e2))
        lat = self.latent(self.down3_4(e3))

        d3 = self.decoder3(self.reduce_chan_level3(torch.cat([self.up4_3(lat), e3], dim=1)))
        d2 = self.decoder2(self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], dim=1)))
        d1 = self.decoder1(torch.cat([self.up2_1(d2), e1], dim=1))
        out = self.output(self.refinement(d1))

        if self.residual:
            return inp[:, : self.out_channels] - out
        return out
