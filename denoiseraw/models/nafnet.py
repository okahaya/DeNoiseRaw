"""NAFNet -- *Simple Baselines for Image Restoration* (Chen et al., ECCV 2022).

The current best accuracy-per-FLOP point on SIDD, and our default. The design
claim is that none of the usual restoration machinery (self-attention, GELU,
complex gating) is necessary: a plain U-Net whose blocks use a multiplicative
gate and a linear channel attention matches or beats transformers at a fraction
of the cost. For 20-50 MP DSLR files that cost difference is the difference
between "usable" and "not".
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .layers import LayerNorm2d, PixelShuffleUp, SimpleGate, SimplifiedChannelAttention


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out: float = 0.0):
        super().__init__()
        dw_c = c * dw_expand
        ffn_c = c * ffn_expand

        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_c, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_c, dw_c, 3, padding=1, groups=dw_c, bias=True)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_c // 2)
        self.conv3 = nn.Conv2d(dw_c // 2, c, 1, bias=True)

        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_c, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_c // 2, c, 1, bias=True)

        self.drop1 = nn.Dropout2d(drop_out) if drop_out > 0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop_out) if drop_out > 0 else nn.Identity()

        # Learnable residual scales, initialised at zero so the block starts as
        # an identity. Deep restoration stacks diverge without this.
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv3(self.sca(self.sg(self.conv2(self.conv1(self.norm1(inp))))))
        y = inp + self.drop1(x) * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + self.drop2(x) * self.gamma


class NAFNet(nn.Module):
    """U-shaped NAFNet.

    Parameters
    ----------
    in_channels / out_channels:
        4 for packed Bayer. ``in_channels`` becomes 8 when a per-pixel sigma map
        is concatenated (see :class:`~denoiseraw.models.wrapper.DenoiserWrapper`).
    width:
        Base channel count. 32 reproduces the paper's "NAFNet-width32"
        (29.2M parameters, 39.97 dB on SIDD); 64 is the large variant.
    residual:
        Predict the noise and subtract it, rather than regressing the clean
        image. Converges faster and keeps highlights numerically exact.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        width: int = 32,
        enc_blocks: Sequence[int] = (2, 2, 4, 8),
        middle_blocks: int = 12,
        dec_blocks: Sequence[int] = (2, 2, 2, 2),
        drop_out: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.out_channels = out_channels

        self.intro = nn.Conv2d(in_channels, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, out_channels, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, drop_out=drop_out) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan, drop_out=drop_out) for _ in range(middle_blocks)])

        for n in dec_blocks:
            self.ups.append(PixelShuffleUp(chan, chan // 2))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, drop_out=drop_out) for _ in range(n)]))

        self.padder_size = 2 ** len(self.encoders)

        if residual:
            _zero_init(self.ending)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.intro(x)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip  # additive skips: half the memory of concatenation
            x = decoder(x)

        x = self.ending(x)
        if self.residual:
            return inp[:, : self.out_channels] - x
        return x


def _zero_init(conv: nn.Module) -> None:
    """Start a residual network as the identity.

    With a zero output layer the first prediction is exactly the input, so the
    network begins from "do nothing" and learns the correction. Without it the
    first few hundred steps are spent unlearning random noise the final layer
    adds, which on a deep stack can be enough to diverge.
    """
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
