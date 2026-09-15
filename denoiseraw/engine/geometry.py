"""Dihedral (D4) transforms for *packed* Bayer data.

Geometric self-ensembling -- denoise all eight flips/rotations and average --
reliably buys 0.1-0.3 dB. On packed Bayer it is also a classic source of silent
bugs, because a spatial flip of the mosaic permutes the CFA:

Take RGGB, whose packed planes are ``R=(0,0)``, ``G1=(0,1)``, ``G2=(1,0)``,
``B=(1,1)``. Flipping the mosaic horizontally makes column 0 of the result the
old column W-1, which is an odd column -- a G1 site. So the flipped mosaic reads
GRBG, and in packed terms ``R`` and ``G1`` have swapped planes (as have ``G2``
and ``B``) on top of the spatial flip.

Ignoring that permutation feeds the network red data on its green input plane.
The ensemble then averages eight mutually inconsistent estimates and *loses*
accuracy -- which looks like "self-ensemble doesn't help here" rather than a bug.

Each generator below is an involution, so a transform is undone by replaying its
generators in reverse order.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# name -> (spatial operation, plane permutation)
_HFLIP_PERM = (1, 0, 3, 2)
_VFLIP_PERM = (2, 3, 0, 1)
_TRANSPOSE_PERM = (0, 2, 1, 3)

GENERATORS = ("transpose", "vflip", "hflip")


def _apply_generator(x, name: str, packed: bool):
    """Apply one generator to ``(..., C, H, W)`` data (numpy array or torch tensor)."""
    is_torch = hasattr(x, "flip")
    if name == "hflip":
        y = x.flip(-1) if is_torch else np.flip(x, axis=-1)
        perm = _HFLIP_PERM
    elif name == "vflip":
        y = x.flip(-2) if is_torch else np.flip(x, axis=-2)
        perm = _VFLIP_PERM
    elif name == "transpose":
        y = x.transpose(-1, -2) if is_torch else np.swapaxes(x, -1, -2)
        perm = _TRANSPOSE_PERM
    else:
        raise ValueError(f"unknown generator {name!r}")

    if not packed:
        return y
    idx = list(perm)
    if is_torch:
        import torch
        return y.index_select(-3, torch.as_tensor(idx, device=y.device))
    return y[..., idx, :, :]


def transform_sequences() -> List[Tuple[str, ...]]:
    """The eight elements of D4, as generator sequences."""
    seqs = []
    for t in (0, 1):
        for v in (0, 1):
            for h in (0, 1):
                seq = tuple(n for n, use in zip(GENERATORS, (t, v, h)) if use)
                seqs.append(seq)
    return seqs


def apply_transform(x, seq: Tuple[str, ...], packed: bool = True):
    """Apply a generator sequence in order."""
    for name in seq:
        x = _apply_generator(x, name, packed)
    return x


def invert_transform(x, seq: Tuple[str, ...], packed: bool = True):
    """Undo :func:`apply_transform`. Generators are involutions, so replay them
    in reverse order."""
    for name in reversed(seq):
        x = _apply_generator(x, name, packed)
    return x
