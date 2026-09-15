"""Image quality metrics.

PSNR alone is a poor guide for denoising -- it rewards over-smoothing -- so a
few complementary measures are provided. The RAW-specific one is
:func:`residual_noise_level`, which asks the question that actually matters:
how much noise is *left*, and is it still consistent with the sensor's noise
model?
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
    out = {
        "residual_noise_before": residual_noise_level(noisy),
        "residual_noise_after": residual_noise_level(denoised),
    }
    before = out["residual_noise_before"]
    if before > 0:
        out["noise_reduction_db"] = float(
            20.0 * np.log10(before / max(out["residual_noise_after"], 1e-12))
        )
    if clean is not None:
        for k, v in evaluate_pair(denoised, clean).items():
            out[k] = v
        out["psnr_input"] = psnr(noisy, clean)
        out["psnr_gain"] = out["psnr"] - out["psnr_input"]
    return out
