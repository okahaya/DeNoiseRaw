"""Training-free RAW denoising: variance stabilisation + a classical filter.

This is the path that works the moment the package is installed, with no
weights and no GPU. The recipe is the standard one for RAW:

1. Pack the mosaic into four colour-consistent planes.
2. Apply the generalised Anscombe transform, so the Poisson-Gaussian noise
   becomes additive with unit variance -- which is what every classical
   denoiser assumes and almost no real sensor provides.
3. Denoise each plane.
4. Invert the transform with the closed-form unbiased inverse.

Quality lands a few dB below a trained network, but it is an honest,
parameter-free baseline and a useful sanity check when a model's output looks
suspicious.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from ..noise.profile import NoiseProfile
from ..vst import gat, gat_inverse

BACKENDS = ("auto", "bm3d", "bm3d-lite", "wavelet", "nlm")


def _have_bm3d() -> bool:
    try:
        import bm3d  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_backend(name: str = "auto") -> str:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")
    if name != "auto":
        return name
    # The `bm3d` package is a mature, tuned implementation and beats our
    # dependency-free one by roughly 0.5 dB, so prefer it when present.
    return "bm3d" if _have_bm3d() else "bm3d-lite"


def denoise_plane_unit_variance(plane: np.ndarray, backend: str = "auto", strength: float = 1.0) -> np.ndarray:
    """Denoise a plane whose noise has already been stabilised to sigma = 1."""
    backend = resolve_backend(backend)
    sigma = float(strength)

    if backend == "bm3d":
        import bm3d
        return bm3d.bm3d(plane.astype(np.float64), sigma).astype(np.float32)

    if backend == "bm3d-lite":
        from .bm3d_lite import bm3d_denoise
        return bm3d_denoise(plane, sigma)

    if backend == "wavelet":
        from skimage.restoration import denoise_wavelet
        return denoise_wavelet(plane, sigma=sigma, mode="soft", method="BayesShrink",
                               wavelet="db4", rescale_sigma=False).astype(np.float32)

    if backend == "nlm":
        from skimage.restoration import denoise_nl_means
        return denoise_nl_means(plane, h=0.8 * sigma, sigma=sigma, patch_size=5,
                                patch_distance=7, fast_mode=True).astype(np.float32)

    raise ValueError(f"unhandled backend {backend!r}")


def denoise_packed(
    packed: np.ndarray,
    profile: NoiseProfile,
    backend: str = "auto",
    strength: float = 1.0,
    progress: bool = False,
) -> np.ndarray:
    """Denoise packed RAW planes using the calibrated noise profile.

    ``strength`` scales the assumed noise level: below 1 keeps more detail (and
    more grain), above 1 smooths harder. 1.0 means "trust the calibration".
    """
    packed = np.asarray(packed, dtype=np.float32)
    single = packed.ndim == 2
    if single:
        packed = packed[None]

    a_arr, b_arr = profile.per_channel_arrays(packed.shape[0])
    out = np.empty_like(packed)

    iterator = range(packed.shape[0])
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="classical denoise", unit="plane")
        except ImportError:
            pass

    for i in iterator:
        a = float(a_arr[i, 0, 0])
        b = float(b_arr[i, 0, 0])
        if a <= 0 and b <= 0:
            out[i] = packed[i]
            continue
        if a <= 1e-9:
            # Pure additive noise: no stabilisation needed, filter directly.
            sigma = float(np.sqrt(max(b, 0.0)))
            out[i] = denoise_plane_unit_variance(packed[i] / sigma, backend, strength) * sigma
            continue
        z = gat(packed[i], a, b)
        z_hat = denoise_plane_unit_variance(z, backend, strength)
        out[i] = gat_inverse(z_hat, a, b)

    return out[0] if single else out


def denoise_raw_image(img, profile: Optional[NoiseProfile] = None, backend: str = "auto",
                      strength: float = 1.0, progress: bool = False):
    """Convenience wrapper operating on a :class:`~denoiseraw.rawio.loader.RawImage`."""
    from ..noise.estimate import estimate_profile
    from ..rawio.packing import pack_bayer, unpack_bayer

    if profile is None:
        profile = estimate_profile(img)
    if img.is_bayer:
        packed = pack_bayer(img.data, img.pattern)
        den = denoise_packed(packed, profile, backend, strength, progress)
        return img.replaced(unpack_bayer(den, img.pattern))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        den = denoise_packed(img.data, profile, backend, strength, progress)
    return img.replaced(den)
