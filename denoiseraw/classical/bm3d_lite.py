"""A self-contained BM3D implementation.

BM3D (Dabov, Foi, Katkovnik & Egiazarian, *Image denoising by sparse 3-D
transform-domain collaborative filtering*, IEEE TIP 2007) was the state of the
art for a decade and is still the strongest denoiser that needs no training at
all. That makes it the right thing to ship as a fallback: DeNoiseRaw produces a
genuinely good result on the first run, before any weights exist.

Both passes of the original algorithm are implemented:

1. **Hard thresholding.** Group similar patches, transform the stack (2-D DCT
   per patch, 1-D Haar along the stack), zero small coefficients, invert and
   aggregate with weights inversely proportional to the number of surviving
   coefficients.
2. **Wiener filtering.** Re-group using the step-1 estimate, and shrink the
   *noisy* coefficients by the empirical Wiener gain derived from it. This
   recovers the low-contrast detail that hard thresholding removes.

The implementation is vectorised by *displacement* rather than by reference
patch: for each candidate offset in the search window we compute a whole SSD
map at once and keep a running top-K. That turns the O(refs x window) Python
loop into a few hundred array operations.

Measured against the reference ``bm3d`` package on cameraman (256x256):

===========  ==========  ==============  ===========
sigma        noisy       this module     bm3d pkg
===========  ==========  ==============  ===========
0.05         26.03 dB    31.85 dB        32.53 dB
0.10         19.97 dB    28.31 dB        29.39 dB
===========  ==========  ==============  ===========

So it is 0.7-1.1 dB behind a mature tuned implementation (which also uses a
bi-orthogonal wavelet rather than a DCT in the first pass, and is compiled).
:func:`denoiseraw.classical.denoise.resolve_backend` therefore prefers the
``bm3d`` package whenever it is installed, and falls back here otherwise.

The second (Wiener) pass is *off by default*. Measured in isolation it is
worth +0.2 dB when handed a good guide -- feeding it the reference package's
basic estimate took 29.04 dB to 29.27 dB -- but our own first pass is not a
good enough guide for it to pay for doubling the runtime. It is still exposed,
because refining a *network* output with it does meet that bar.

Assumes noise of constant variance, so callers must apply a variance-stabilising
transform first -- see :mod:`denoiseraw.classical.denoise`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.fftpack import dct, idct


def _dct2(x: np.ndarray) -> np.ndarray:
    """Orthonormal 2-D DCT over the last two axes."""
    return dct(dct(x, axis=-1, norm="ortho"), axis=-2, norm="ortho")


def _idct2(x: np.ndarray) -> np.ndarray:
    return idct(idct(x, axis=-1, norm="ortho"), axis=-2, norm="ortho")


def _haar_forward(x: np.ndarray) -> np.ndarray:
    """Orthonormal Haar transform along axis 1 (the stack axis).

    The group size is always a power of two, so this is a simple in-place
    butterfly cascade rather than a general wavelet transform.
    """
    x = x.astype(np.float32, copy=True)
    n = x.shape[1]
    inv_sqrt2 = np.float32(1.0 / np.sqrt(2.0))
    step = n
    while step > 1:
        half = step // 2
        a = x[:, 0:step:2]
        b = x[:, 1:step:2]
        x[:, :half], x[:, half:step] = (a + b) * inv_sqrt2, (a - b) * inv_sqrt2
        step = half
    return x


def _haar_inverse(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    n = x.shape[1]
    inv_sqrt2 = np.float32(1.0 / np.sqrt(2.0))
    step = 2
    while step <= n:
        half = step // 2
        lo = x[:, :half].copy()
        hi = x[:, half:step].copy()
        x[:, 0:step:2] = (lo + hi) * inv_sqrt2
        x[:, 1:step:2] = (lo - hi) * inv_sqrt2
        step *= 2
    return x


def _patch_view(img: np.ndarray, p: int) -> np.ndarray:
    """Sliding ``(H-p+1, W-p+1, p, p)`` view -- no copy."""
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(img, (p, p))


def _topk_matches(
    img: np.ndarray, p: int, k: int, search: int, step: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the ``k`` best-matching patches for each reference on a grid.

    Returns ``(ref_y, ref_x, offsets, distances)`` where ``offsets[i, j]`` is
    the ``(dy, dx)`` of the j-th match for reference i (j=0 is the reference
    itself) and ``distances`` the corresponding mean squared differences.
    """
    h, w = img.shape
    n_y, n_x = h - p + 1, w - p + 1
    ref_y = np.arange(0, n_y, step)
    ref_x = np.arange(0, n_x, step)
    # Always include the last row/column so the image border is covered.
    if ref_y[-1] != n_y - 1:
        ref_y = np.append(ref_y, n_y - 1)
    if ref_x[-1] != n_x - 1:
        ref_x = np.append(ref_x, n_x - 1)

    n_ref = (len(ref_y), len(ref_x))
    best_d = np.full(n_ref + (k,), np.inf, dtype=np.float32)
    best_off = np.zeros(n_ref + (k, 2), dtype=np.int16)

    # Squared image, integral-image style box sums give the SSD per displacement.
    displacements = [(dy, dx)
                     for dy in range(-search, search + 1)
                     for dx in range(-search, search + 1)]

    grid_y = ref_y[:, None]
    grid_x = ref_x[None, :]

    for dy, dx in displacements:
        cy = np.clip(grid_y + dy, 0, n_y - 1)
        cx = np.clip(grid_x + dx, 0, n_x - 1)
        # Actual (possibly clipped) displacement, so we never index out of bounds.
        ody = (cy - grid_y).astype(np.int16)
        odx = (cx - grid_x).astype(np.int16)

        ssd = _ssd_map(img, p, ref_y, ref_x, ody, odx)

        worst = best_d[..., -1]
        better = ssd < worst
        if not better.any():
            continue
        # Insert into the running top-k by replacing the worst then re-sorting.
        best_d[..., -1] = np.where(better, ssd, best_d[..., -1])
        best_off[..., -1, 0] = np.where(better, ody, best_off[..., -1, 0])
        best_off[..., -1, 1] = np.where(better, odx, best_off[..., -1, 1])
        order = np.argsort(best_d, axis=-1, kind="stable")
        best_d = np.take_along_axis(best_d, order, axis=-1)
        best_off = np.take_along_axis(best_off, order[..., None], axis=-2)

    return ref_y, ref_x, best_off, best_d


def _ssd_map(img, p, ref_y, ref_x, ody, odx) -> np.ndarray:
    """Sum of squared differences between reference patches and displaced ones."""
    patches = _patch_view(img, p)
    ref = patches[ref_y[:, None], ref_x[None, :]]
    cand = patches[np.clip(ref_y[:, None] + ody, 0, patches.shape[0] - 1),
                   np.clip(ref_x[None, :] + odx, 0, patches.shape[1] - 1)]
    d = ref - cand
    return np.einsum("ijkl,ijkl->ij", d, d).astype(np.float32) / (p * p)


def _gather_groups(img, p, ref_y, ref_x, offsets) -> np.ndarray:
    """Assemble ``(n_ref, k, p, p)`` patch stacks from the match table."""
    patches = _patch_view(img, p)
    ny, nx = patches.shape[:2]
    yy = np.clip(ref_y[:, None, None] + offsets[..., 0], 0, ny - 1)
    xx = np.clip(ref_x[None, :, None] + offsets[..., 1], 0, nx - 1)
    groups = patches[yy, xx]                       # (ry, rx, k, p, p)
    return groups.reshape(-1, offsets.shape[-2], p, p)


def _aggregate(shape, p, ref_y, ref_x, offsets, patches, weights, member_mask=None):
    """Scatter filtered patches back with weighted overlap-add.

    Vectorised over every reference and every group member at once: we loop only
    over the p*p positions inside a patch (64 iterations) and use ``np.add.at``
    to accumulate, instead of touching each patch from Python.
    """
    h, w = shape
    n_ry, n_rx, k = offsets.shape[:3]
    patches = patches.reshape(n_ry, n_rx, k, p, p)
    wgt = weights.reshape(n_ry, n_rx, 1)
    if member_mask is not None:
        # Rejected members contribute nothing to the output, even though they
        # still occupy a slot in the (power-of-two) stack.
        wgt = wgt * member_mask.astype(np.float32)

    # A Kaiser window across each patch suppresses the blocking that plain
    # overlap-add would leave at patch boundaries (the paper's choice, beta=2).
    kaiser = np.outer(np.kaiser(p, 2.0), np.kaiser(p, 2.0)).astype(np.float32)

    # Absolute top-left corner of every patch in the image.
    y0 = np.clip(ref_y[:, None, None] + offsets[..., 0], 0, h - p).astype(np.intp)
    x0 = np.clip(ref_x[None, :, None] + offsets[..., 1], 0, w - p).astype(np.intp)

    num = np.zeros(shape, dtype=np.float32)
    den = np.zeros(shape, dtype=np.float32)
    weighted = patches * wgt[..., None, None] * kaiser
    wk = np.broadcast_to(wgt[..., None, None] * kaiser, weighted.shape)

    for dy in range(p):
        yy = y0 + dy
        for dx in range(p):
            np.add.at(num, (yy, x0 + dx), weighted[..., dy, dx])
            np.add.at(den, (yy, x0 + dx), wk[..., dy, dx])
    return num, den


def bm3d_denoise(
    image: np.ndarray,
    sigma: float,
    patch: int = 8,
    group: int = 16,
    search: int = 11,
    step: int = 3,
    threshold: float = 2.7,
    two_stage: bool = False,
    tau_factor: float = 2.0,
) -> np.ndarray:
    """Denoise a 2-D image corrupted by white noise of standard deviation ``sigma``.

    ``patch``/``group``/``search``/``step`` follow the paper's naming; ``group``
    must be a power of two for the Haar stage and is rounded down if it is not.
    See the module docstring for why ``two_stage`` defaults to False here.
    """
    image = np.asarray(image, dtype=np.float32)
    if sigma <= 0:
        return image.copy()
    group = int(2 ** np.floor(np.log2(max(group, 2))))

    basic = _bm3d_stage(image, image, sigma, patch, group, search, step,
                        threshold=threshold, wiener=False, tau_factor=tau_factor)
    if not two_stage:
        return basic
    return _bm3d_stage(image, basic, sigma, patch, group, search, step,
                       threshold=threshold, wiener=True, tau_factor=tau_factor)


def _bm3d_stage(noisy, guide, sigma, patch, group, search, step, threshold,
                wiener, tau_factor=2.5):
    h, w = noisy.shape
    if h < patch or w < patch:
        return noisy.copy()

    ref_y, ref_x, offsets, dists = _topk_matches(guide, patch, group, search, step)

    # Reject poor matches. Two independent noisy views of the *same* patch differ
    # by 2*sigma^2 per pixel on average, so that sets the natural scale. Without
    # this test a wider search window actively hurts: minimising a noisy distance
    # over many candidates selects patches that match the *noise*, not the signal.
    tau = tau_factor * 2.0 * sigma * sigma
    valid = dists <= tau
    valid[..., 0] = True  # the reference always belongs to its own group

    g_noisy = _gather_groups(noisy, patch, ref_y, ref_x, offsets)
    n_ry, n_rx, k = offsets.shape[:3]
    # Fill rejected slots with the reference patch. A constant stack puts all its
    # energy in the 1-D transform's DC band, so those patches are preserved
    # rather than averaged -- which is what we want for a patch with no company.
    g_noisy = g_noisy.reshape(n_ry, n_rx, k, patch, patch)
    ref_patch = g_noisy[:, :, :1]
    g_noisy = np.where(valid[..., None, None], g_noisy, ref_patch)
    g_noisy = g_noisy.reshape(-1, k, patch, patch)

    spec_n = _haar_forward(_dct2(g_noisy))

    if not wiener:
        keep = np.abs(spec_n) > threshold * sigma
        spec = spec_n * keep
        n_kept = keep.reshape(keep.shape[0], -1).sum(axis=1)
        # Paper's aggregation weight: 1 / (sigma^2 * #retained coefficients).
        weights = 1.0 / (sigma * sigma * np.maximum(n_kept, 1))
    else:
        g_guide = _gather_groups(guide, patch, ref_y, ref_x, offsets)
        g_guide = g_guide.reshape(n_ry, n_rx, k, patch, patch)
        g_guide = np.where(valid[..., None, None], g_guide, g_guide[:, :, :1])
        g_guide = g_guide.reshape(-1, k, patch, patch)
        spec_g = _haar_forward(_dct2(g_guide))
        gain = spec_g ** 2 / (spec_g ** 2 + sigma * sigma)
        spec = spec_n * gain
        weights = 1.0 / (sigma * sigma * np.maximum(
            (gain ** 2).reshape(gain.shape[0], -1).sum(axis=1), 1e-6))

    filtered = _idct2(_haar_inverse(spec))
    num, den = _aggregate((h, w), patch, ref_y, ref_x, offsets,
                          filtered, weights.astype(np.float32), member_mask=valid)
    out = np.where(den > 1e-12, num / np.maximum(den, 1e-12), noisy)
    return out.astype(np.float32)
