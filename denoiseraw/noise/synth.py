"""Synthesising realistic sensor noise.

Training a RAW denoiser needs (clean, noisy) pairs. Capturing them is expensive
and covers only the ISOs you happened to shoot, so the standard recipe --
*Unprocessing* (Brooks et al., CVPR 2019), ELD (Wei et al., CVPR 2020) and
*Rethinking Noise Synthesis* (Zhang et al., ICCV 2021) all agree on this -- is to
take clean low-ISO RAW frames and inject noise sampled from a calibrated model.

Two models are provided:

``poisson_gaussian``
    Shot noise as a true Poisson draw plus a Gaussian read term. Accurate in
    normal light and the right default for daylight DSLR work.

``eld``
    The full physics model: Poisson shot noise, Tukey-lambda read noise, per-row
    banding and the quantiser. This is what makes a denoiser survive deep
    shadows and high ISO, where the read distribution's tails dominate and
    banding is the artefact people actually complain about.

Noise is always injected in the *packed* domain (4 planes) so that row noise is
applied to genuine sensor rows: in a Bayer mosaic, two packed planes share each
sensor row pair, which the :func:`add_row_noise` helper accounts for.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .profile import NoiseProfile, tukey_lambda_std


def _rng(rng) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def sample_tukey_lambda(shape, lam: float, sigma: float, rng=None) -> np.ndarray:
    """Draw zero-mean Tukey-lambda noise with standard deviation ``sigma``.

    Uses the distribution's closed-form quantile function, which is why this is
    cheap despite having no simple density.
    """
    g = _rng(rng)
    u = g.uniform(1e-6, 1.0 - 1e-6, size=shape)
    if abs(lam) < 1e-6:
        z = np.log(u / (1.0 - u))
    else:
        z = (np.power(u, lam) - np.power(1.0 - u, lam)) / lam
    return (z / tukey_lambda_std(lam) * sigma).astype(np.float32)


def add_shot_noise(clean: np.ndarray, a: np.ndarray, rng=None) -> np.ndarray:
    """Photon shot noise: ``x -> a * Poisson(x / a)``.

    ``a`` is the variance slope, i.e. one "electron" is worth ``a`` in normalised
    units. A true Poisson draw (rather than a Gaussian with matching variance)
    matters in deep shadows, where the electron count is single-digit and the
    distribution is visibly skewed.

    Values below zero are passed through untouched rather than clamped. After
    black-level subtraction real RAW data does go slightly negative, and those
    pixels collected no photons, so there is no shot noise to add -- clamping
    them to zero would erase the below-black tail and bake a colour cast into
    the shadows of anything trained on the result.
    """
    g = _rng(rng)
    a = np.asarray(a, dtype=np.float32)
    safe_a = np.maximum(a, 1e-12)
    positive = np.maximum(clean, 0.0)
    offset = clean - positive          # zero, except where clean < 0
    lam = positive / safe_a
    # Poisson sampling gets slow and overflows at huge rates; above ~1e6 electrons
    # the Gaussian limit is exact to far more precision than 16 bits can hold.
    shot = np.where(
        lam > 1e6,
        positive + g.standard_normal(clean.shape).astype(np.float32) * np.sqrt(safe_a * positive),
        g.poisson(np.minimum(lam, 1e6)).astype(np.float32) * safe_a,
    )
    return (shot + offset).astype(np.float32)


def add_row_noise(x: np.ndarray, sigma_row: float, packed: bool = True, rng=None) -> np.ndarray:
    """Per-sensor-row banding.

    Read-out amplifiers are shared along a row, so their fluctuation is a single
    offset applied to every pixel in that row. On packed Bayer data each packed
    row corresponds to one physical sensor row, but planes R/G1 come from the
    *same* physical row (and G2/B from the next), so they must share an offset —
    getting this wrong turns coherent banding into incoherent per-plane noise and
    the model never learns to remove real banding.
    """
    if sigma_row <= 0:
        return x
    g = _rng(rng)
    if packed and x.ndim == 3 and x.shape[0] == 4:
        h = x.shape[1]
        # Two physical rows per packed row: (R, G1) then (G2, B).
        offsets = g.standard_normal((2, h, 1)).astype(np.float32) * sigma_row
        rows = np.stack([offsets[0], offsets[0], offsets[1], offsets[1]], axis=0)
        return x + rows
    offsets = g.standard_normal(x.shape[:-1] + (1,)).astype(np.float32) * sigma_row
    return x + offsets


def add_quantization(x: np.ndarray, step: float, rng=None) -> np.ndarray:
    """Uniform quantisation noise of one ADU."""
    if step <= 0:
        return x
    g = _rng(rng)
    return x + g.uniform(-0.5, 0.5, size=x.shape).astype(np.float32) * step


def synthesize_noise(
    clean: np.ndarray,
    profile: NoiseProfile,
    model: str = "eld",
    packed: bool = True,
    clip: Optional[Tuple[float, float]] = None,
    rng=None,
) -> np.ndarray:
    """Add sensor noise to a clean normalised RAW array.

    Parameters
    ----------
    clean:
        ``(4, H, W)`` packed planes (``packed=True``) or ``(H, W)`` mosaic.
    profile:
        Calibrated :class:`NoiseProfile`.
    model:
        ``"eld"``, ``"poisson_gaussian"`` or ``"gaussian"``.
    clip:
        Optional ``(lo, hi)`` clamp. Leave ``None`` during training: clipping the
        below-black tail is the single most common way to bake a colour cast into
        a denoiser's shadows.
    """
    g = _rng(rng)
    clean = np.asarray(clean, dtype=np.float32)

    if packed and clean.ndim == 3 and clean.shape[0] == 4:
        a_arr, b_arr = profile.per_channel_arrays(4)
    else:
        a_arr = np.float32(profile.a)
        b_arr = np.float32(profile.b)

    if model == "gaussian":
        sigma = np.sqrt(np.maximum(b_arr, 0.0) + np.maximum(a_arr, 0.0) * np.maximum(clean, 0.0))
        noisy = clean + g.standard_normal(clean.shape).astype(np.float32) * sigma

    elif model == "poisson_gaussian":
        noisy = add_shot_noise(clean, a_arr, rng=g)
        noisy = noisy + g.standard_normal(clean.shape).astype(np.float32) * np.sqrt(np.maximum(b_arr, 0.0))

    elif model == "eld":
        noisy = add_shot_noise(clean, a_arr, rng=g)
        # Split the signal-independent budget: row banding is drawn separately,
        # so the Tukey term only carries what is left of the measured variance.
        b_scalar = float(np.mean(b_arr))
        row_var = min(profile.sigma_row ** 2, 0.9 * b_scalar) if profile.sigma_row > 0 else 0.0
        quant_var = (profile.quantization_step ** 2) / 12.0 if profile.quantization_step > 0 else 0.0
        read_var = max(b_scalar - row_var - quant_var, 0.0)
        if read_var > 0:
            noisy = noisy + sample_tukey_lambda(
                clean.shape, profile.tukey_lambda, float(np.sqrt(read_var)), rng=g
            )
        if row_var > 0:
            noisy = add_row_noise(noisy, float(np.sqrt(row_var)), packed=packed, rng=g)
        noisy = add_quantization(noisy, profile.quantization_step, rng=g)

    else:
        raise ValueError(f"unknown noise model {model!r}")

    if clip is not None:
        noisy = np.clip(noisy, clip[0], clip[1])
    return noisy.astype(np.float32)


def sample_profile(
    bank_or_profile,
    iso_range: Tuple[float, float] = (100.0, 25600.0),
    jitter_db: float = 1.0,
    rng=None,
) -> NoiseProfile:
    """Draw a random profile for one training sample.

    Gains are sampled uniformly in log space (ELD's scheme) and then perturbed,
    because no calibration is exact and a denoiser that has only ever seen the
    nominal parameters is brittle when the real body deviates.
    """
    from .profile import ProfileBank

    g = _rng(rng)
    iso = float(np.exp(g.uniform(np.log(iso_range[0]), np.log(iso_range[1]))))
    base = bank_or_profile.at_iso(iso) if isinstance(bank_or_profile, ProfileBank) else bank_or_profile

    jitter = float(10.0 ** (g.uniform(-jitter_db, jitter_db) / 20.0))
    out = NoiseProfile.from_dict(base.to_dict())
    out.a = base.a * jitter
    out.b = base.b * jitter * jitter
    out.sigma_row = base.sigma_row * jitter
    out.a_per_channel = [v * jitter for v in base.a_per_channel]
    out.b_per_channel = [v * jitter * jitter for v in base.b_per_channel]
    # Read-noise shape varies between bodies and ISOs; sample it too.
    out.tukey_lambda = float(np.clip(g.normal(base.tukey_lambda, 0.15), -0.4, 0.6))
    out.iso = iso
    out.source = "sampled"
    return out
