"""Synthetic RAW fixtures.

The test suite cannot ship a 30 MB camera file, so it manufactures one: a
scene with the frequency content real photographs have, mosaicked to a Bayer
pattern, corrupted with the physics model, and written as a DNG that libraw
opens exactly like a camera's own file.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from denoiseraw.noise.profile import NoiseProfile
from denoiseraw.noise.synth import synthesize_noise
from denoiseraw.rawio.loader import RawImage
from denoiseraw.rawio.packing import pack_bayer, unpack_bayer

# A real Canon 5D Mark IV cam_xyz, so colour handling is exercised with
# plausible numbers rather than the identity.
CANON_5D4_CAM_XYZ = np.array([
    [0.6446, -0.0366, -0.0864],
    [-0.4436, 1.2204, 0.2513],
    [-0.0952, 0.1873, 0.6607],
    [0.0, 0.0, 0.0],
])


def synthetic_scene(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    """An ``(H, W, 3)`` linear RGB scene with edges, texture and smooth gradients."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Smooth illumination gradient.
    base = 0.08 + 0.35 * (xx / w) + 0.15 * (yy / h)
    scene = np.stack([base * 1.05, base, base * 0.92], axis=-1)

    # Hard-edged patches: the structure a denoiser must not blur.
    for _ in range(18):
        cy, cx = int(rng.integers(0, h - 40)), int(rng.integers(0, w - 40))
        sh, sw = int(rng.integers(12, 40)), int(rng.integers(12, 40))
        scene[cy:cy + sh, cx:cx + sw] += rng.uniform(-0.15, 0.25, size=3)

    # Fine near-Nyquist texture, in patches -- which is how real photographs are
    # built. A globally textured frame would leave the blind noise estimator no
    # flat area to measure, which is a real limitation but not the common case.
    texture = 0.06 * np.sin(xx / 2.5) * np.cos(yy / 3.1)
    for _ in range(6):
        cy, cx = int(rng.integers(0, h - 60)), int(rng.integers(0, w - 60))
        sh, sw = int(rng.integers(30, 60)), int(rng.integers(30, 60))
        scene[cy:cy + sh, cx:cx + sw] += texture[cy:cy + sh, cx:cx + sw, None]
    # A bright specular region, to exercise highlight behaviour.
    scene[h // 8: h // 8 + 24, w // 8: w // 8 + 24] += 0.45
    return np.clip(scene, 0.002, 0.98).astype(np.float32)


def mosaic_from_rgb(rgb: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Sample an RGB image onto a Bayer grid."""
    lut = {"R": 0, "G": 1, "B": 2}
    offsets = ((0, 0), (0, 1), (1, 0), (1, 1))
    mosaic = np.zeros(rgb.shape[:2], dtype=np.float32)
    for letter, (dy, dx) in zip(pattern, offsets):
        mosaic[dy::2, dx::2] = rgb[dy::2, dx::2, lut[letter]]
    return mosaic


def make_raw_image(
    h: int = 256,
    w: int = 256,
    pattern: str = "RGGB",
    profile: Optional[NoiseProfile] = None,
    seed: int = 0,
    noise_model: str = "eld",
    white_level: float = 16383.0,
    black_level: float = 512.0,
) -> Tuple[RawImage, np.ndarray, NoiseProfile]:
    """Return ``(noisy RawImage, clean mosaic, profile)``."""
    scene = synthetic_scene(h, w, seed)
    clean = mosaic_from_rgb(scene, pattern)
    profile = profile or NoiseProfile(
        a=3e-4, b=1.6e-5, sigma_row=1.2e-3, tukey_lambda=0.14,
        quantization_step=1.0 / (white_level - black_level),
        camera_model="Canon EOS 5D Mark IV", iso=3200,
        scale=white_level - black_level,
    )
    packed = pack_bayer(clean, pattern)
    noisy = unpack_bayer(
        synthesize_noise(packed, profile, model=noise_model,
                         rng=np.random.default_rng(seed + 1)),
        pattern,
    )
    img = RawImage(
        data=noisy.astype(np.float32),
        pattern=pattern,
        black_level=black_level,
        white_level=white_level,
        black_level_per_channel=[black_level] * 4,
        camera_whitebalance=[2.08, 1.0, 1.52, 1.0],
        daylight_whitebalance=[2.0, 1.0, 1.5, 1.0],
        color_matrix=CANON_5D4_CAM_XYZ,
        camera_model="Canon EOS 5D Mark IV",
        iso=3200,
        exposure_time=1 / 125,
        aperture=2.8,
    )
    return img, clean, profile


def write_synthetic_dng(path: str, **kwargs) -> Tuple[str, np.ndarray, NoiseProfile]:
    """Write a synthetic noisy capture as a DNG libraw can open."""
    from denoiseraw.rawio.writer import save_dng

    img, clean, profile = make_raw_image(**kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    save_dng(path, img)
    return path, clean, profile
