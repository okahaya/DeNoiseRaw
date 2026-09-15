"""A small separable-convolution U-Net for fast / CPU-only denoising.

Follows the shape of PMRID (Wang et al., *Practical Deep Raw Image Denoising on
Mobile Devices*, ECCV 2020): an encoder-decoder built from depth-wise separable
blocks, sized so a full-resolution 24 MP frame is minutes rather than hours on a
laptop CPU. It gives up roughly 0.5-0.8 dB against NAFNet, which is a trade most
people will take when the alternative is no GPU at all.
"""

from __future__ import annotations

from torch import nn


class SeparableBlock(nn.Module):
    """Depth-wise 5x5 + point-wise 1x1, ~8x cheaper than a dense 5x5."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 5, stride=stride, padding=2, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class LiteUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        width: int = 32,
        depth: int = 4,
        blocks_per_level: int = 2,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.out_channels = out_channels

        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        chans = []
        for _ in range(depth):
            self.encoders.append(nn.Sequential(*[SeparableBlock(chan, chan) for _ in range(blocks_per_level)]))
            chans.append(chan)
            self.downs.append(SeparableBlock(chan, min(chan * 2, 256), stride=2))
            chan = min(chan * 2, 256)

        self.middle = nn.Sequential(*[SeparableBlock(chan, chan) for _ in range(blocks_per_level)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for skip_c in reversed(chans):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, skip_c * 4, 1), nn.PixelShuffle(2)))
            self.decoders.append(nn.Sequential(*[SeparableBlock(skip_c, skip_c) for _ in range(blocks_per_level)]))
            chan = skip_c

        self.ending = nn.Conv2d(chan, out_channels, 3, padding=1)
        self.padder_size = 2 ** depth

        if residual:
            # Start as the identity; see NAFNet's _zero_init for why.
            nn.init.zeros_(self.ending.weight)
            nn.init.zeros_(self.ending.bias)

    def forward(self, x):
        inp = x
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x) + skip
            x = dec(x)
        x = self.ending(x)
        return inp[:, : self.out_channels] - x if self.residual else x
