"""End-to-end RAW denoising: file in, denoised file out.

This is the layer the CLI and any GUI sit on. It handles the decisions a user
should not have to make by hand:

* where the noise parameters come from (a calibration file if you have one,
  otherwise estimated from the image itself);
* whether to run the neural model or the classical fallback;
* tiling and memory;
* and what to write out.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .noise.estimate import estimate_profile
from .noise.profile import NoiseProfile, ProfileBank
from .rawio.develop import develop
from .rawio.loader import RawImage, load_raw
from .rawio.packing import pack_bayer, unpack_bayer

METHODS = ("auto", "model", "classical", "none")


@dataclass
class DenoiseSettings:
    """Everything that controls a denoising run."""

    method: str = "auto"
    checkpoint: Optional[str] = None
    preset: str = "nafnet"
    strength: float = 1.0
    classical_backend: str = "auto"

    # Noise parameters
    profile_path: Optional[str] = None
    iso: Optional[float] = None
    estimate_noise: bool = True

    # Inference
    tile: int = 512
    overlap: int = 64
    self_ensemble: bool = False
    device: str = "auto"
    amp: bool = True

    # Output
    develop_preview: bool = True
    demosaic: str = "malvar"
    exposure: float = 0.0
    auto_bright: bool = False
    progress: bool = True

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DenoiseResult:
    raw: RawImage
    profile: NoiseProfile
    method: str
    elapsed: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def preview(self, settings: Optional[DenoiseSettings] = None) -> np.ndarray:
        s = settings or DenoiseSettings()
        return develop(self.raw, method=s.demosaic, exposure=s.exposure,
                       auto_bright=s.auto_bright)


def resolve_profile(img: RawImage, settings: DenoiseSettings) -> NoiseProfile:
    """Decide which noise parameters to use, in order of trustworthiness.

    A calibration measured from your own camera beats anything estimated, so it
    wins when supplied. Otherwise we fit the Poisson-Gaussian law to this very
    frame, which is accurate to a few percent on anything with flat areas.
    """
    if settings.profile_path:
        path = settings.profile_path
        try:
            bank = ProfileBank.load(path)
            if bank.profiles:
                iso = settings.iso or img.iso or 100.0
                return bank.at_iso(float(iso))
        except (KeyError, ValueError, TypeError):
            pass
        return NoiseProfile.load(path)

    if not settings.estimate_noise:
        raise ValueError("no noise profile supplied and estimation is disabled")
    return estimate_profile(img)


def choose_method(settings: DenoiseSettings) -> str:
    """Resolve ``auto``: use the network if we have weights, else classical."""
    if settings.method != "auto":
        return settings.method
    if settings.checkpoint and os.path.exists(settings.checkpoint):
        return "model"
    return "classical"


def denoise_packed_array(packed: np.ndarray, profile: NoiseProfile,
                         settings: DenoiseSettings) -> np.ndarray:
    """Denoise packed planes with whichever backend the settings select."""
    method = choose_method(settings)

    if method == "none":
        return packed

    if method == "classical":
        from .classical import denoise_packed as classical_denoise
        return classical_denoise(packed, profile, backend=settings.classical_backend,
                                 strength=settings.strength, progress=settings.progress)

    from .engine.infer import denoise_packed as model_denoise
    from .models import load_checkpoint

    if settings.checkpoint and os.path.exists(settings.checkpoint):
        model, _ = load_checkpoint(settings.checkpoint)
    else:
        # An untrained network is worse than no denoising at all, so make the
        # failure loud rather than returning a plausible-looking wrong answer.
        raise FileNotFoundError(
            f"method='model' needs trained weights; checkpoint {settings.checkpoint!r} not found. "
            "Train one with `denoiseraw train`, or use --method classical."
        )

    expected = getattr(model, "channels", 4)
    if expected != packed.shape[0]:
        kind = "Bayer (4 packed planes)" if packed.shape[0] == 4 else \
            f"non-Bayer ({packed.shape[0]} plane)"
        raise ValueError(
            f"checkpoint was trained for {expected}-channel data but this file is {kind}. "
            "Bayer and non-Bayer (X-Trans, Quad-Bayer) sensors need separate models; "
            "train one with `denoiseraw train` on files from the same kind of sensor."
        )

    if settings.strength != 1.0:
        # The model is conditioned on the noise level, so asking it to clean
        # more or less is just a matter of overstating or understating sigma.
        profile = NoiseProfile.from_dict({
            **profile.to_dict(),
            "a": profile.a * settings.strength ** 2,
            "b": profile.b * settings.strength ** 2,
            "a_per_channel": [v * settings.strength ** 2 for v in profile.a_per_channel],
            "b_per_channel": [v * settings.strength ** 2 for v in profile.b_per_channel],
        })

    return model_denoise(model, packed, profile, tile=settings.tile, overlap=settings.overlap,
                         self_ensemble=settings.self_ensemble, amp=settings.amp,
                         device=settings.device, progress=settings.progress)


def denoise_raw(img: RawImage, settings: Optional[DenoiseSettings] = None,
                profile: Optional[NoiseProfile] = None) -> DenoiseResult:
    """Denoise an already-loaded :class:`RawImage`."""
    settings = settings or DenoiseSettings()
    start = time.time()
    profile = profile or resolve_profile(img, settings)

    if img.is_bayer:
        packed = pack_bayer(img.data, img.pattern)
        denoised = denoise_packed_array(packed, profile, settings)
        out = img.replaced(unpack_bayer(denoised, img.pattern))
    else:
        denoised = denoise_packed_array(img.data[None], profile, settings)
        out = img.replaced(denoised[0])

    from .metrics import residual_noise_level

    metrics = {
        "residual_noise_before": residual_noise_level(
            pack_bayer(img.data, img.pattern) if img.is_bayer else img.data[None]),
        "residual_noise_after": residual_noise_level(
            pack_bayer(out.data, out.pattern) if out.is_bayer else out.data[None]),
    }
    if metrics["residual_noise_after"] > 0:
        metrics["noise_reduction_db"] = float(
            20.0 * np.log10(metrics["residual_noise_before"] / metrics["residual_noise_after"]))

    return DenoiseResult(raw=out, profile=profile, method=choose_method(settings),
                         elapsed=time.time() - start, metrics=metrics)


def denoise_file(path: str, settings: Optional[DenoiseSettings] = None) -> DenoiseResult:
    """Load a RAW file, denoise it, and return the result."""
    settings = settings or DenoiseSettings()
    img = load_raw(path)
    return denoise_raw(img, settings)


def write_outputs(result: DenoiseResult, output: str,
                  settings: Optional[DenoiseSettings] = None) -> Dict[str, str]:
    """Write the result, choosing the format from ``output``'s extension."""
    from .rawio.writer import save_dng, save_image, save_linear_cfa

    settings = settings or DenoiseSettings()
    ext = os.path.splitext(output)[1].lower()
    written: Dict[str, str] = {}

    if ext == ".dng":
        written["dng"] = save_dng(output, result.raw)
    elif ext in (".tif", ".tiff") and settings.extra.get("linear_cfa"):
        written["cfa"] = save_linear_cfa(output, result.raw)
    elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        written["image"] = save_image(output, result.preview(settings))
    else:
        raise ValueError(f"unsupported output extension {ext!r}")
    return written
