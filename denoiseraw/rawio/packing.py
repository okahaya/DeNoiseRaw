"""CFA <-> tensor packing.

Networks want dense, spatially-aligned feature planes. A Bayer mosaic is none of
those things: neighbouring pixels see different colours, so a plain convolution
mixes channels that have wildly different statistics. Every modern RAW denoiser
(SID, PMRID, ELD, CycleISP, Unprocessing) therefore *packs* the 2x2 tile into
four half-resolution planes before the first convolution and unpacks afterwards.

Two conventions matter and are easy to get wrong:

* The pack order must follow the sensor's CFA pattern, otherwise a model trained
  on RGGB silently degrades on a BGGR body.
* Packing must be exactly invertible, including for odd-sized sensors, so we
  crop to an even size once and remember the crop.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Where each of the four packed planes sits inside the 2x2 tile, per CFA pattern.
# Planes are emitted in canonical R, G1, G2, B order so a model sees the same
# semantics regardless of the body it was fed.
_TILE_OFFSETS = {
    #            R       G1      G2      B
    "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "BGGR": ((1, 1), (0, 1), (1, 0), (0, 0)),
    "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "GBRG": ((1, 0), (0, 0), (1, 1), (0, 1)),
}

PACKED_CHANNEL_NAMES = ("R", "G1", "G2", "B")


def even_crop(arr: np.ndarray) -> np.ndarray:
    """Crop the trailing row/column so both spatial dims are even."""
    h, w = arr.shape[-2:]
    return arr[..., : h - (h % 2), : w - (w % 2)]


def pack_bayer(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    """``(H, W)`` mosaic -> ``(4, H/2, W/2)`` planes ordered R, G1, G2, B."""
    if pattern not in _TILE_OFFSETS:
        raise ValueError(f"unsupported CFA pattern {pattern!r}; expected one of {list(_TILE_OFFSETS)}")
    mosaic = even_crop(np.asarray(mosaic))
    planes = [mosaic[dy::2, dx::2] for dy, dx in _TILE_OFFSETS[pattern]]
    return np.stack(planes, axis=0)


def unpack_bayer(packed: np.ndarray, pattern: str) -> np.ndarray:
    """Inverse of :func:`pack_bayer`."""
    if pattern not in _TILE_OFFSETS:
        raise ValueError(f"unsupported CFA pattern {pattern!r}")
    packed = np.asarray(packed)
    if packed.shape[0] != 4:
        raise ValueError(f"expected 4 packed planes, got {packed.shape[0]}")
    _, h, w = packed.shape
    mosaic = np.empty((h * 2, w * 2), dtype=packed.dtype)
    for plane, (dy, dx) in zip(packed, _TILE_OFFSETS[pattern]):
        mosaic[dy::2, dx::2] = plane
    return mosaic


def cfa_color_index(pattern: str) -> np.ndarray:
    """Return a ``(2, 2)`` array mapping tile position -> 0=R, 1=G, 2=B."""
    lut = {"R": 0, "G": 1, "B": 2}
    return np.array([[lut[pattern[0]], lut[pattern[1]]],
                     [lut[pattern[2]], lut[pattern[3]]]], dtype=np.int64)


def channel_color_index(pattern: str) -> np.ndarray:
    """Colour (0=R, 1=G, 2=B) of each of the four packed planes.

    Always ``[0, 1, 1, 2]`` by construction, but kept explicit so callers that
    broadcast per-colour noise parameters read clearly.
    """
    cfa = cfa_color_index(pattern)
    return np.array([cfa[dy, dx] for dy, dx in _TILE_OFFSETS[pattern]], dtype=np.int64)


def pad_to_multiple(arr: np.ndarray, multiple: int, mode: str = "reflect") -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad the last two dims up to a multiple of ``multiple`` (for U-Net depth)."""
    h, w = arr.shape[-2:]
    ph = (-h) % multiple
    pw = (-w) % multiple
    if ph == 0 and pw == 0:
        return arr, (0, 0)
    pad = [(0, 0)] * (arr.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(arr, pad, mode=mode), (ph, pw)


def unpad(arr: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
    ph, pw = pad
    h, w = arr.shape[-2:]
    return arr[..., : h - ph if ph else h, : w - pw if pw else w]
