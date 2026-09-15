"""Noise calibration from bias and flat-field frames.

This is the accurate route, and it is what ELD's per-camera parameters come
from. Two kinds of frames are needed, both trivially shootable:

**Bias / dark frames** — lens cap on, fastest shutter, at the ISO you care
about. These contain *only* signal-independent noise, so they give ``b``, the
row-noise sigma, and the shape of the read-noise distribution directly.

**Flat frames** — an evenly lit featureless surface (a defocused white wall, a
lightbox) shot as a bracketed series from near-black to near-clipping at the
same ISO. Because the scene is constant, the variance between two consecutive
exposures of the same brightness isolates the shot noise, which gives ``a``.

The two-frame differencing trick matters: subtracting two frames of the same
scene cancels fixed-pattern noise (PRNU, dark current structure), which is
*not* what we want the denoiser to remove — it is stable and correctable
elsewhere in the pipeline, and treating it as noise makes the model blur
genuine detail.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..rawio.loader import RawImage, load_raw
from ..rawio.packing import pack_bayer
from .profile import NoiseProfile, ProfileBank


def _packed(img: RawImage) -> np.ndarray:
    return pack_bayer(img.data, img.pattern) if img.is_bayer else img.data[None]


def fit_tukey_lambda(samples: np.ndarray, candidates: Optional[Sequence[float]] = None) -> float:
    """Pick the Tukey-lambda shape whose quantiles best match the data.

    A probability-plot correlation search: for each candidate lambda, compare
    the sample quantiles against the theoretical ones and keep the best straight
    line. This is the standard way to fit Tukey lambda, which has no closed-form
    density to maximise.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 256:
        return 0.14
    if x.size > 200_000:
        x = np.random.default_rng(0).choice(x, 200_000, replace=False)
    x = np.sort(x)
    n = x.size
    p = (np.arange(1, n + 1) - 0.3175) / (n + 0.365)  # Filliben plotting positions
    if candidates is None:
        candidates = np.linspace(-0.4, 0.6, 51)

    best_lam, best_r = 0.14, -np.inf
    for lam in candidates:
        if abs(lam) < 1e-6:
            q = np.log(p / (1.0 - p))
        else:
            q = (np.power(p, lam) - np.power(1.0 - p, lam)) / lam
        r = float(np.corrcoef(q, x)[0, 1])
        if r > best_r:
            best_r, best_lam = r, float(lam)
    return best_lam


def calibrate_from_bias(
    bias_frames: Sequence[RawImage],
    fit_lambda: bool = True,
) -> Tuple[float, float, float]:
    """Return ``(b, sigma_row, tukey_lambda)`` from dark frames.

    With two or more frames we difference consecutive pairs to cancel the fixed
    pattern; with one frame we fall back to a robust spread of the frame itself,
    which slightly overestimates ``b`` by the amount of fixed-pattern noise.
    """
    if not bias_frames:
        raise ValueError("no bias frames supplied")

    planes = [_packed(f) for f in bias_frames]
    shape = planes[0].shape
    planes = [p for p in planes if p.shape == shape]

    if len(planes) >= 2:
        # Var(A - B) = 2 * Var(read); the fixed pattern cancels.
        diffs = [(planes[i] - planes[i + 1]) / np.sqrt(2.0) for i in range(len(planes) - 1)]
        residual = np.concatenate([d.ravel() for d in diffs])
        row_src = list(diffs)
    else:
        residual = (planes[0] - np.median(planes[0])).ravel()
        row_src = [planes[0] - np.median(planes[0])]

    mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    b_total = float(mad ** 2)

    # Row noise: the spread of row means, corrected for the pixel noise that
    # survives averaging over a row.
    row_sigmas = []
    for src in row_src:
        for plane in src:
            w = plane.shape[1]
            rows = plane.mean(axis=1)
            rm = 1.4826 * np.median(np.abs(rows - np.median(rows)))
            row_sigmas.append(np.sqrt(max(rm * rm - b_total / max(w, 1), 0.0)))
    sigma_row = float(np.median(row_sigmas)) if row_sigmas else 0.0

    # Caveat: when frames were differenced, `residual` is a *difference* of two
    # read-noise draws, which is closer to Gaussian than either one (central
    # limit). The fitted lambda is therefore biased towards 0.14 -- treat it as a
    # lower bound on the tail weight, and prefer a single-frame fit if the exact
    # shape matters more than the fixed-pattern cancellation.
    lam = fit_tukey_lambda(residual) if fit_lambda else 0.14
    return b_total, sigma_row, lam


def calibrate_from_flats(
    flat_pairs: Iterable[Tuple[RawImage, RawImage]],
    b: float = 0.0,
    n_bins: int = 24,
) -> float:
    """Return the shot-noise slope ``a`` from pairs of identical flat exposures.

    Each pair contributes ``Var(A - B) / 2`` at signal level ``(A + B) / 2``;
    regressing that against the level gives ``a`` with the fixed pattern removed.
    """
    levels: List[float] = []
    variances: List[float] = []

    for fa, fb in flat_pairs:
        pa, pb = _packed(fa), _packed(fb)
        if pa.shape != pb.shape:
            continue
        diff = (pa - pb) / np.sqrt(2.0)
        mean = (pa + pb) / 2.0
        # Bin by signal level so a slightly uneven flat still contributes cleanly.
        lo, hi = float(np.percentile(mean, 1)), float(np.percentile(mean, 99))
        if hi - lo < 1e-6:
            lo, hi = float(mean.min()), float(mean.max()) + 1e-6
        edges = np.linspace(lo, hi, n_bins + 1)
        idx = np.clip(np.digitize(mean, edges) - 1, 0, n_bins - 1)
        for k in range(n_bins):
            sel = idx == k
            if sel.sum() < 512:
                continue
            d = diff[sel]
            mad = 1.4826 * np.median(np.abs(d - np.median(d)))
            levels.append(float(mean[sel].mean()))
            variances.append(float(mad ** 2))

    if len(levels) < 2:
        raise ValueError("not enough usable flat-field samples; shoot a wider exposure ramp")

    x = np.asarray(levels)
    y = np.asarray(variances) - b  # remove the already-known read floor
    # Weight by level: the bright end constrains the slope, the dark end is noisy.
    w = np.sqrt(np.maximum(x, 1e-6))
    a = float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-12))
    return max(a, 0.0)


def calibrate(
    bias_paths: Sequence[str],
    flat_paths: Sequence[str] = (),
    iso: Optional[float] = None,
) -> NoiseProfile:
    """Full calibration from files on disk.

    ``flat_paths`` should be an exposure ramp with *pairs* of identical shots
    (shoot each level twice); consecutive files are paired in order.
    """
    bias = [load_raw(p) for p in bias_paths]
    b, sigma_row, lam = calibrate_from_bias(bias)

    ref = bias[0]
    a = 0.0
    if len(flat_paths) >= 2:
        flats = [load_raw(p) for p in flat_paths]
        pairs = [(flats[i], flats[i + 1]) for i in range(0, len(flats) - 1, 2)]
        a = calibrate_from_flats(pairs, b=b)

    return NoiseProfile(
        a=a,
        b=b,
        sigma_row=sigma_row,
        tukey_lambda=lam,
        quantization_step=1.0 / ref.scale if ref.scale else 0.0,
        camera_model=ref.camera_model,
        iso=iso if iso is not None else ref.iso,
        scale=ref.scale,
        source="calibrated",
    )


def calibrate_bank(jobs: Sequence[dict]) -> ProfileBank:
    """Calibrate several ISOs at once.

    Each job is ``{"iso": 1600, "bias": [...], "flats": [...]}``.
    """
    bank = ProfileBank()
    for job in jobs:
        prof = calibrate(job.get("bias", []), job.get("flats", []), iso=job.get("iso"))
        bank.camera_model = prof.camera_model
        bank.add(prof)
    return bank
