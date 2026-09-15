"""Blind noise estimation from a single RAW frame.

Calibration frames are the accurate route, but nobody shoots bias frames before
a holiday. So we also estimate the heteroscedastic law ``Var(x) = a x + b``
directly from the image the user handed us, following the scatter-plot approach
of Foi et al., *Practical Poissonian-Gaussian noise modeling and fitting for
single-image raw-data* (IEEE TIP 2008).

The method in three steps:

1. Split each packed plane into small blocks and estimate a local mean and a
   local noise variance. Variance comes from a high-pass filter whose response
   to white noise has a known gain (Immerkaer 1996), so texture contributes far
   less than it would to a raw sample variance.
2. Keep only blocks that look flat. Textured blocks inflate the variance and
   would bias ``b`` upwards — visibly over-smoothing the result.
3. Fit a line through the (mean, variance) scatter with an iteratively
   reweighted robust fit, since even after step 2 some blocks contain structure
   and those outliers all point the same way.

Accuracy and its limits
-----------------------
On ordinary photographs -- which contain *some* smooth area, be it sky, a wall
or a shadow -- this recovers the shot-noise slope to within a few percent. It
degrades on frames that are textured at fine scale *everywhere* (dense foliage,
fabric, gravel filling the frame): a high-pass filter cannot tell near-Nyquist
detail from noise, so the estimate biases ``b`` up and ``a`` down, and the
result is over-smoothed. If that matters, calibrate the camera once with
:mod:`denoiseraw.noise.calibrate` and pass the profile explicitly; measured
parameters do not care what the scene looks like.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..rawio.packing import pack_bayer
from .profile import NoiseProfile

# Immerkaer's 3x3 high-pass mask. For zero-mean white noise of variance v, the
# filtered signal has variance 36 v, while a linear ramp passes through as zero.
_IMMERKAER = np.array([[1.0, -2.0, 1.0],
                       [-2.0, 4.0, -2.0],
                       [1.0, -2.0, 1.0]], dtype=np.float32)
_IMMERKAER_GAIN = 36.0


def _highpass(plane: np.ndarray) -> np.ndarray:
    from scipy.signal import convolve2d

    return convolve2d(plane, _IMMERKAER, mode="valid")


def _block_reduce(arr: np.ndarray, block: int, func) -> np.ndarray:
    h, w = arr.shape
    bh, bw = h // block, w // block
    if bh == 0 or bw == 0:
        return np.empty((0,), dtype=arr.dtype)
    trimmed = arr[: bh * block, : bw * block].reshape(bh, block, bw, block)
    return func(trimmed, axis=(1, 3))


def _mad_variance(blocks: np.ndarray, axis) -> np.ndarray:
    """Median-absolute-deviation variance estimate, robust to outlier pixels."""
    med = np.median(blocks, axis=axis, keepdims=True)
    mad = np.median(np.abs(blocks - med), axis=axis)
    return (mad * 1.4826) ** 2


def plane_scatter(plane: np.ndarray, block: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-block (mean, noise variance, flatness score) for one packed plane."""
    plane = np.asarray(plane, dtype=np.float32)
    hp = _highpass(plane)
    # Align the high-pass result with the original by trimming the 1px border.
    core = plane[1:-1, 1:-1]
    n = min(core.shape[0], hp.shape[0]), min(core.shape[1], hp.shape[1])
    core, hp = core[: n[0], : n[1]], hp[: n[0], : n[1]]

    means = _block_reduce(core, block, np.mean)
    var = _block_reduce(hp, block, _mad_variance) / _IMMERKAER_GAIN
    # Flatness: spread of the block's own pixel values relative to its noise.
    # A flat block's spread is explained by noise alone; a textured one is not.
    spread = _block_reduce(core, block, np.std)
    flatness = spread / np.sqrt(np.maximum(var, 1e-12))
    return means.ravel(), var.ravel(), flatness.ravel()


def _robust_linear_fit(x: np.ndarray, y: np.ndarray, iters: int = 12) -> Tuple[float, float]:
    """IRLS line fit with Tukey biweight, resistant to one-sided outliers."""
    if len(x) < 4:
        return 0.0, float(np.median(y)) if len(y) else 0.0
    X = np.stack([x, np.ones_like(x)], axis=1)
    w = np.ones_like(x)
    a = b = 0.0
    for _ in range(iters):
        W = w[:, None]
        try:
            coef, *_ = np.linalg.lstsq(X * W, y * w, rcond=None)
        except np.linalg.LinAlgError:
            break
        a, b = float(coef[0]), float(coef[1])
        resid = y - (a * x + b)
        s = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-12
        u = resid / (4.685 * s)
        w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        if w.sum() < 4:
            break
    return a, b


def estimate_profile(
    raw_or_array,
    pattern: Optional[str] = None,
    block: int = 24,
    flat_quantile: float = 0.35,
    scale: Optional[float] = None,
) -> NoiseProfile:
    """Estimate ``Var(x) = a x + b`` from a single normalised RAW frame.

    Parameters
    ----------
    raw_or_array:
        A :class:`~denoiseraw.rawio.loader.RawImage`, a ``(H, W)`` mosaic (then
        ``pattern`` is required) or an already-packed ``(4, H, W)`` array.
    flat_quantile:
        Fraction of blocks, ranked by flatness, kept for the fit. Lower is more
        conservative (fewer but cleaner samples).
    """
    from ..rawio.loader import RawImage

    camera, iso = "unknown", None
    if isinstance(raw_or_array, RawImage):
        img = raw_or_array
        camera, iso = img.camera_model, img.iso
        scale = scale if scale is not None else img.scale
        if img.is_bayer:
            packed = pack_bayer(img.data, img.pattern)
        else:
            packed = img.data[None]
    else:
        arr = np.asarray(raw_or_array, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] in (1, 4):
            packed = arr
        elif arr.ndim == 2:
            if pattern is None:
                raise ValueError("a CFA pattern is required to pack a 2-D mosaic")
            packed = pack_bayer(arr, pattern)
        else:
            raise ValueError(f"cannot interpret array of shape {arr.shape}")

    means, vars_ = [], []
    per_channel = []
    for plane in packed:
        m, v, f = plane_scatter(plane, block=block)
        keep = np.isfinite(m) & np.isfinite(v) & np.isfinite(f) & (v > 0)
        m, v, f = m[keep], v[keep], f[keep]
        if len(m) >= 8:
            thr = np.quantile(f, flat_quantile)
            sel = f <= thr
            m, v = m[sel], v[sel]
        per_channel.append(_robust_linear_fit(m, v))
        means.append(m)
        vars_.append(v)

    m_all = np.concatenate(means) if means else np.zeros(0)
    v_all = np.concatenate(vars_) if vars_ else np.zeros(0)
    a, b = _robust_linear_fit(m_all, v_all)

    # Negative parameters are physically impossible and come from a degenerate
    # fit (e.g. an almost uniform frame). Fall back to the variance floor.
    a = max(a, 0.0)
    if b <= 0:
        b = float(np.percentile(v_all, 10)) if len(v_all) else 1e-8
    b = max(b, 1e-12)

    prof = NoiseProfile(
        a=float(a),
        b=float(b),
        a_per_channel=[max(ai, 0.0) for ai, _ in per_channel] if len(per_channel) == 4 else [],
        b_per_channel=[max(bi, 1e-12) for _, bi in per_channel] if len(per_channel) == 4 else [],
        quantization_step=(1.0 / scale) if scale else 0.0,
        camera_model=camera,
        iso=iso,
        scale=scale,
        source="blind-estimate",
    )
    # Hand the row-noise estimator the per-pixel noise variance we just fitted,
    # so it can separate banding from ordinary pixel noise.
    plane_means = np.array([float(np.mean(np.clip(pl, 0, None))) for pl in packed])
    a_pc = np.asarray(prof.a_per_channel) if prof.a_per_channel else np.full(len(packed), a)
    b_pc = np.asarray(prof.b_per_channel) if prof.b_per_channel else np.full(len(packed), b)
    prof.sigma_row = estimate_row_noise(packed, pixel_var=a_pc * plane_means + b_pc)
    return prof


def estimate_row_noise(packed: np.ndarray, pixel_var: Optional[np.ndarray] = None,
                       max_rows: int = 2048) -> float:
    """Estimate the banding component from row-mean statistics.

    Averaging a row of ``W`` pixels suppresses per-pixel noise by ``sqrt(W)`` but
    leaves the row's shared offset untouched, so the row means are dominated by
    banding. Differencing consecutive row means removes the image's own vertical
    gradients, and a MAD makes the estimate robust to the rows that cross an
    edge.

    ``pixel_var`` is the per-pixel *noise* variance, which the caller normally
    already knows from the fitted ``a x + b``. It must not be measured from the
    image's own vertical differences: on a real photograph those are dominated by
    scene structure, which overestimates the term and drives the result to zero.
    """
    packed = np.asarray(packed, dtype=np.float32)
    if packed.ndim != 3:
        return 0.0
    pv = np.zeros(packed.shape[0]) if pixel_var is None else np.broadcast_to(
        np.asarray(pixel_var, dtype=np.float64).reshape(-1), (packed.shape[0],))

    ests = []
    for plane, plane_var in zip(packed, pv):
        w = plane.shape[1]
        rows = plane[:max_rows].mean(axis=1)
        if len(rows) < 16:
            continue
        d = np.diff(rows)
        mad = 1.4826 * np.median(np.abs(d - np.median(d)))
        # Var(diff of row means) = 2 * (row_var + pixel_var / W).
        row_var = max(mad * mad / 2.0 - float(plane_var) / max(w, 1), 0.0)
        ests.append(np.sqrt(row_var))
    return float(np.median(ests)) if ests else 0.0
