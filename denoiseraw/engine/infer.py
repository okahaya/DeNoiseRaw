"""Whole-image inference.

A 45 MP RAW packs to four 3000x4000 planes. Feeding that to a network in one
piece needs tens of gigabytes of activations, so we tile. Tiling done naively
leaves visible seams, because each tile's border pixels have no context on one
side; the fixes are to overlap tiles and to cross-fade them with a window that
sums to one everywhere.

Also here: optional geometric self-ensembling (see
:mod:`denoiseraw.engine.geometry`), which averages the model's output over the
eight flips and rotations of the input for a small but free quality gain.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from ..noise.profile import NoiseProfile
from .geometry import apply_transform, invert_transform, transform_sequences


def pick_device(preference: str = "auto") -> torch.device:
    """Resolve ``auto`` to the best device available."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _blend_window(h: int, w: int, overlap: int, dtype, device) -> torch.Tensor:
    """A separable window that ramps up over ``overlap`` pixels at each edge.

    Tiles are placed on a grid whose stride is ``tile - overlap``, so exactly two
    tiles cover each overlap band and their ramps are mirror images. A linear
    (Bartlett) ramp therefore sums to 1 across the seam -- we still normalise by
    the accumulated weight afterwards, which also handles the image borders and
    the last, unevenly-placed row and column of tiles.
    """
    def ramp(n: int) -> torch.Tensor:
        win = torch.ones(n, dtype=dtype, device=device)
        k = min(overlap, n // 2)
        if k > 0:
            # The ramp starts just above zero rather than at zero. A tile's very
            # outermost pixels are computed from reflect padding instead of real
            # neighbours, so they are the least trustworthy and get the smallest
            # weight -- but at the image border no other tile covers them, and a
            # weight of exactly zero would leave the denominator at zero there.
            edge = torch.linspace(1.0 / (k + 1), k / (k + 1), k, dtype=dtype, device=device)
            win[:k] = edge
            win[-k:] = edge.flip(0)
        return win

    return ramp(h)[:, None] * ramp(w)[None, :]


@torch.no_grad()
def denoise_packed_tensor(
    model: torch.nn.Module,
    packed: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tile: int = 512,
    overlap: int = 64,
    self_ensemble: bool = False,
    amp: bool = True,
    device: Optional[torch.device] = None,
    progress: bool = False,
) -> torch.Tensor:
    """Denoise ``(C, H, W)`` packed planes with overlap-blended tiling.

    ``a`` and ``b`` are ``(C,)`` or ``(C, 1, 1)`` Poisson-Gaussian parameters.
    """
    device = device or next(model.parameters()).device
    model.eval()

    packed = packed.to(device=device, dtype=torch.float32)
    c, h, w = packed.shape
    a = a.reshape(1, -1, 1, 1).to(device=device, dtype=torch.float32)
    b = b.reshape(1, -1, 1, 1).to(device=device, dtype=torch.float32)

    use_amp = amp and device.type == "cuda"
    seqs = transform_sequences() if self_ensemble else [()]

    if tile <= 0 or (h <= tile and w <= tile):
        positions = [(0, 0, h, w)]
    else:
        stride = max(tile - overlap, 1)
        ys = list(range(0, max(h - tile, 0) + 1, stride))
        xs = list(range(0, max(w - tile, 0) + 1, stride))
        if ys[-1] != max(h - tile, 0):
            ys.append(max(h - tile, 0))
        if xs[-1] != max(w - tile, 0):
            xs.append(max(w - tile, 0))
        positions = [(y, x, min(tile, h), min(tile, w)) for y in ys for x in xs]

    numerator = torch.zeros((c, h, w), dtype=torch.float32, device=device)
    denominator = torch.zeros((1, h, w), dtype=torch.float32, device=device)

    iterator = positions
    if progress and len(positions) > 1:
        try:
            from tqdm import tqdm
            iterator = tqdm(positions, desc="denoise", unit="tile")
        except ImportError:
            pass

    for (y, x, th, tw) in iterator:
        patch = packed[:, y:y + th, x:x + tw].unsqueeze(0)
        acc = torch.zeros_like(patch)
        for seq in seqs:
            inp = apply_transform(patch, seq, packed=True) if seq else patch
            # The noise parameters are per-plane, so they must follow the same
            # plane permutation the geometry applied.
            a_t = apply_transform(a.expand(1, c, 1, 1), seq, packed=True) if seq else a
            b_t = apply_transform(b.expand(1, c, 1, 1), seq, packed=True) if seq else b
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(inp.contiguous(), a_t, b_t)
            out = out.float()
            acc += invert_transform(out, seq, packed=True) if seq else out
        acc /= len(seqs)

        win = _blend_window(th, tw, overlap, torch.float32, device)
        numerator[:, y:y + th, x:x + tw] += acc[0] * win
        denominator[:, y:y + th, x:x + tw] += win

    return numerator / denominator.clamp(min=1e-8)


def denoise_packed(
    model: torch.nn.Module,
    packed: np.ndarray,
    profile: NoiseProfile,
    tile: int = 512,
    overlap: int = 64,
    self_ensemble: bool = False,
    amp: bool = True,
    device: str = "auto",
    progress: bool = False,
) -> np.ndarray:
    """NumPy-facing wrapper around :func:`denoise_packed_tensor`."""
    dev = pick_device(device)
    model = model.to(dev)
    packed = np.asarray(packed, dtype=np.float32)
    single = packed.ndim == 2
    if single:
        packed = packed[None]

    a_arr, b_arr = profile.per_channel_arrays(packed.shape[0])
    out = denoise_packed_tensor(
        model,
        torch.from_numpy(packed),
        torch.from_numpy(a_arr.reshape(-1)),
        torch.from_numpy(b_arr.reshape(-1)),
        tile=tile, overlap=overlap, self_ensemble=self_ensemble,
        amp=amp, device=dev, progress=progress,
    ).cpu().numpy()
    return out[0] if single else out


def estimate_tile_size(free_bytes: Optional[int] = None, channels: int = 8,
                       safety: float = 0.25) -> int:
    """Pick a tile size that should fit in available GPU memory.

    Rough but useful: activation memory scales with tile area times the width of
    the network, so we budget a fixed fraction of free memory and round down to a
    multiple of 64.
    """
    if free_bytes is None:
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
        else:
            free_bytes = 2 << 30
    budget = free_bytes * safety
    # ~200 bytes of activations per input pixel per input channel, measured on
    # NAFNet-width32; conservative for the smaller presets.
    px = budget / (200.0 * channels)
    side = int(math.sqrt(max(px, 64 * 64)))
    return max(128, (side // 64) * 64)
