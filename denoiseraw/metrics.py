"""Image quality metrics.

PSNR alone is a poor guide for denoising -- it rewards over-smoothing -- so a
few complementary measures are provided. The RAW-specific one is
:func:`residual_noise_level`, which asks the question that actually matters:
how much noise is *left*, and is it still consistent with the sensor's noise
model?

Comparing two independent :func:`residual_noise_level` calls before/after is
tempting but unsound, and it fails on real photographs, not just in theory: an
aggressive denoiser can push part of the image to near-perfect flatness (dead
shadows, over-smoothed sky), and the *after* image's own "most flat" block
selection then latches onto those artefacts as its best evidence, reporting a
noise level near zero and a reduction in the tens of dB that the image does
not remotely show. Verified on a real ISO 3200 frame: two independent classical
backends both reported 30+ dB "noise reduction" this way, while the actual
pixel-level change in matched, real flat regions was under 15%.
:func:`paired_noise_reduction` fixes this by anchoring the flat-block
selection to the *noisy* input, so both measurements look at the same real
scene regions instead of each image separately nominating its own.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((np.asarray(pred, np.float64) - np.asarray(target, np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    from skimage.metrics import structural_similarity

    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    channel_axis = 0 if pred.ndim == 3 and pred.shape[0] <= 4 else (-1 if pred.ndim == 3 else None)
    return float(structural_similarity(target, pred, data_range=data_range,
                                       channel_axis=channel_axis))


def residual_noise_level(image: np.ndarray, block: int = 24) -> float:
    """Estimate the standard deviation of whatever noise remains.

    Reuses the blind estimator's flat-block statistics. Comparing this before
    and after denoising says how much noise was removed without needing a clean
    reference -- useful on real photographs, where there is none.
    """
    from .noise.estimate import plane_scatter

    planes = image if image.ndim == 3 else image[None]
    sigmas = []
    for plane in planes:
        _, var, flat = plane_scatter(np.asarray(plane, np.float32), block=block)
        ok = np.isfinite(var) & np.isfinite(flat) & (var > 0)
        if ok.sum() >= 8:
            thr = np.quantile(flat[ok], 0.3)
            sigmas.append(float(np.sqrt(np.median(var[ok][flat[ok] <= thr]))))
    return float(np.median(sigmas)) if sigmas else 0.0


def paired_noise_reduction(before: np.ndarray, after: np.ndarray,
                           block: int = 24, flat_quantile: float = 0.3) -> Dict[str, float]:
    """Noise level before/after, measured on the *same* real-scene regions.

    Selects the flattest 30% of blocks using only the noisy ``before`` image's
    own texture, then measures both images' noise variance at those same block
    positions. This is what makes the comparison fair: the alternative of
    picking each image's own flattest blocks independently rewards a denoiser
    for destroying detail into flatness, not for removing noise.
    """
    from .noise.estimate import _IMMERKAER_GAIN, _block_reduce, _highpass, _mad_variance

    before_planes = np.asarray(before, np.float32)
    after_planes = np.asarray(after, np.float32)
    before_planes = before_planes if before_planes.ndim == 3 else before_planes[None]
    after_planes = after_planes if after_planes.ndim == 3 else after_planes[None]
    if before_planes.shape != after_planes.shape:
        raise ValueError(f"shape mismatch: before={before_planes.shape} after={after_planes.shape}")

    sigmas_before, sigmas_after = [], []
    for plane_b, plane_a in zip(before_planes, after_planes):
        hp_b, hp_a = _highpass(plane_b), _highpass(plane_a)
        core = plane_b[1:-1, 1:-1]
        n = (min(core.shape[0], hp_b.shape[0]), min(core.shape[1], hp_b.shape[1]))
        core, hp_b, hp_a = core[:n[0], :n[1]], hp_b[:n[0], :n[1]], hp_a[:n[0], :n[1]]

        var_b = _block_reduce(hp_b, block, _mad_variance) / _IMMERKAER_GAIN
        var_a = _block_reduce(hp_a, block, _mad_variance) / _IMMERKAER_GAIN
        spread = _block_reduce(core, block, np.std)
        flat = (spread / np.sqrt(np.maximum(var_b, 1e-12))).ravel()
        var_b, var_a = var_b.ravel(), var_a.ravel()

        ok = np.isfinite(var_b) & np.isfinite(flat) & (var_b > 0)
        var_b, var_a, flat = var_b[ok], var_a[ok], flat[ok]
        if len(flat) < 8:
            continue
        # The selection mask comes entirely from `before`; `after` is only ever
        # read at the positions that mask picks, never used to pick them.
        selected = flat <= np.quantile(flat, flat_quantile)
        sigmas_before.append(float(np.sqrt(np.median(var_b[selected]))))
        sigmas_after.append(float(np.sqrt(np.median(np.maximum(var_a[selected], 0.0)))))

    sigma_before = float(np.median(sigmas_before)) if sigmas_before else 0.0
    sigma_after = float(np.median(sigmas_after)) if sigmas_after else 0.0
    out = {"residual_noise_before": sigma_before, "residual_noise_after": sigma_after}
    if sigma_before > 0:
        out["noise_reduction_db"] = float(20.0 * np.log10(sigma_before / max(sigma_after, 1e-12)))
    return out


def detail_preservation(pred: np.ndarray, target: np.ndarray) -> float:
    """Ratio of high-frequency energy retained, 1.0 meaning "identical detail".

    Below 1 means detail was smoothed away; meaningfully above 1 means the
    output was sharpened or has artefacts. Denoisers that win on PSNR by
    blurring score poorly here, which is the point.
    """
    from scipy.ndimage import gaussian_filter

    def hf_energy(x: np.ndarray) -> float:
        x = np.asarray(x, np.float64)
        return float(np.var(x - gaussian_filter(x, 2.0)))

    t = hf_energy(target)
    return float(np.sqrt(hf_energy(pred) / t)) if t > 0 else 0.0


def evaluate_pair(pred: np.ndarray, target: np.ndarray,
                  data_range: float = 1.0) -> Dict[str, float]:
    """All reference metrics at once."""
    return {
        "psnr": psnr(pred, target, data_range),
        "ssim": ssim(pred, target, data_range),
        "detail_preservation": detail_preservation(pred, target),
    }


def report(noisy: np.ndarray, denoised: np.ndarray,
           clean: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Summarise a denoising run, with or without ground truth."""
    out = paired_noise_reduction(noisy, denoised)
    if clean is not None:
        for k, v in evaluate_pair(denoised, clean).items():
            out[k] = v
        out["psnr_input"] = psnr(noisy, clean)
        out["psnr_gain"] = out["psnr"] - out["psnr_input"]
    return out
