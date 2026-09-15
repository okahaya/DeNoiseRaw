"""Variance-stabilising transforms, for NumPy arrays and Torch tensors alike.

Poisson-Gaussian noise is *heteroscedastic*: bright pixels are noisier than dark
ones. Classical denoisers (BM3D, wavelet shrinkage) assume constant-variance
noise, so they need the signal remapped first. The generalised Anscombe
transform (GAT) does exactly that::

    f(x) = (2 / a) * sqrt(a*x + (3/8)*a^2 + b)

after which the noise has approximately unit variance everywhere.

Inverting it is the subtle part. The algebraic inverse is *biased*, because
``sqrt`` is concave and the denoiser returns an estimate of ``E[f(x)]`` rather
than ``f(E[x])``. Makitalo & Foi (*A closed-form approximation of the exact
unbiased inverse of the Anscombe variance-stabilizing transformation*, IEEE TIP
2011) give a closed form that corrects this; it is worth several tenths of a dB
in the shadows, which is precisely where RAW denoising is judged.

The functions here are written against operations both NumPy and Torch provide,
so one implementation serves the classical path and the network path.
"""

from __future__ import annotations

import math

SQRT_3_2 = math.sqrt(1.5)


def _sqrt(x):
    return x ** 0.5


def _clamp(x, lo):
    # Works for both numpy arrays and torch tensors without importing either.
    try:
        return x.clamp(min=lo)          # torch
    except (AttributeError, TypeError):
        return x.clip(lo)               # numpy


def gat(x, a, b):
    """Generalised Anscombe transform of a signal with ``Var = a*x + b``."""
    inner = _clamp(a * x + 0.375 * a * a + b, 0.0)
    return (2.0 / a) * _sqrt(inner)


def gat_inverse_algebraic(z, a, b):
    """Exact algebraic inverse of :func:`gat` (biased for denoised input)."""
    return a * (z * z / 4.0 - 0.375) - b / a


def gat_inverse(z, a, b):
    """Closed-form approximate *unbiased* inverse (Makitalo & Foi, 2011).

    The GAT reduces to a plain Anscombe transform of ``u = x/a + b/a^2``, so we
    apply the published closed-form Anscombe inverse to ``z`` and map back.
    """
    # Guard the negative powers: D -> 0 happens in deep shadows and would blow up.
    d = _clamp(z, 0.8)
    inv_u = (
        d * d / 4.0
        + 0.25 * SQRT_3_2 / d
        - 1.375 / (d * d)
        + 0.625 * SQRT_3_2 / (d * d * d)
        - 0.125
    )
    return a * inv_u - b / a


def stabilise(x, a, b):
    """Convenience alias mirroring the paper's naming."""
    return gat(x, a, b)
