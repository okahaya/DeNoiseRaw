"""Developing denoised RAW data into a viewable image.

Denoising happens in linear sensor space, but nobody looks at a Bayer mosaic. To
judge the result -- and to hand people a file they can use -- we need the rest of
the pipeline: demosaic, white balance, the camera's colour matrix, and a tone
curve.

This is deliberately a *reference* development, not a raw converter. It exists so
you can see what the denoiser did. For final output, most people will want to
denoise, write the result back as a linear CFA file, and develop it in
Lightroom / Capture One / darktable with their own settings.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .loader import RawImage
from .packing import cfa_color_index

# sRGB (D65) primaries -> CIE XYZ. The standard matrix; dcraw calls it xyz_rgb.
XYZ_RGB = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

DEMOSAIC_METHODS = ("malvar", "bilinear")

# OpenCV's edge-aware and VNG demosaic paths are deliberately not offered: they
# assert 8-bit input, and quantising linear 14-bit sensor data to 8 bits before
# interpolation throws away exactly the shadow precision this package exists to
# preserve. Malvar below is float throughout and measures better than OpenCV's
# 16-bit-capable path (plain bilinear) anyway.

# Malvar, He & Cutler, "High-quality linear interpolation for demosaicing of
# Bayer-patterned color images" (ICASSP 2004). Gradient-corrected bilinear:
# cheap, and visibly better than bilinear on the fine detail RAW denoising is
# judged on.
_K_G_AT_RB = np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 2, 0, 0],
    [-1, 2, 4, 2, -1],
    [0, 0, 2, 0, 0],
    [0, 0, -1, 0, 0],
], dtype=np.float64) / 8.0

_K_RB_AT_G_ROW = np.array([
    [0, 0, 0.5, 0, 0],
    [0, -1, 0, -1, 0],
    [-1, 4, 5, 4, -1],
    [0, -1, 0, -1, 0],
    [0, 0, 0.5, 0, 0],
], dtype=np.float64) / 8.0

_K_RB_AT_G_COL = _K_RB_AT_G_ROW.T

_K_RB_AT_BR = np.array([
    [0, 0, -1.5, 0, 0],
    [0, 2, 0, 2, 0],
    [-1.5, 0, 6, 0, -1.5],
    [0, 2, 0, 2, 0],
    [0, 0, -1.5, 0, 0],
], dtype=np.float64) / 8.0


def _conv(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from scipy.signal import convolve2d

    return convolve2d(img, kernel, mode="same", boundary="symm")


def demosaic(mosaic: np.ndarray, pattern: str, method: str = "malvar") -> np.ndarray:
    """Interpolate a Bayer mosaic to ``(H, W, 3)`` linear RGB."""
    if method not in DEMOSAIC_METHODS:
        raise ValueError(f"unknown demosaic method {method!r}; expected one of {DEMOSAIC_METHODS}")
    mosaic = np.asarray(mosaic, dtype=np.float32)

    if method == "bilinear":
        return _demosaic_bilinear(mosaic, pattern)
    return _demosaic_malvar(mosaic, pattern)


def _cfa_masks(shape: Tuple[int, int], pattern: str):
    """Boolean masks selecting the R, G and B sites of the mosaic."""
    h, w = shape
    idx = cfa_color_index(pattern)
    yy, xx = np.mgrid[0:h, 0:w]
    site = idx[yy % 2, xx % 2]
    return site == 0, site == 1, site == 2


def _demosaic_bilinear(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    mask_r, mask_g, mask_b = _cfa_masks(mosaic.shape, pattern)
    k_rb = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    k_g = np.array([[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]])
    out = np.stack([
        _conv(mosaic * mask_r, k_rb),
        _conv(mosaic * mask_g, k_g),
        _conv(mosaic * mask_b, k_rb),
    ], axis=-1)
    return out.astype(np.float32)


def _demosaic_malvar(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    mask_r, mask_g, mask_b = _cfa_masks(mosaic.shape, pattern)
    m = mosaic.astype(np.float64)

    g_at_rb = _conv(m, _K_G_AT_RB)
    rb_at_g_row = _conv(m, _K_RB_AT_G_ROW)
    rb_at_g_col = _conv(m, _K_RB_AT_G_COL)
    rb_at_br = _conv(m, _K_RB_AT_BR)

    green = np.where(mask_g, m, g_at_rb)

    # A green site sits either in a red row or a blue row; the correct kernel
    # depends on which, so build per-site masks for the two cases.
    idx = cfa_color_index(pattern)
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    row_colour = idx[yy % 2, (xx % 2 + 1) % 2]   # colour of the horizontal neighbour
    g_in_red_row = mask_g & (row_colour == 0)
    g_in_blue_row = mask_g & (row_colour == 2)

    red = np.where(mask_r, m, 0.0)
    red = np.where(g_in_red_row, rb_at_g_row, red)
    red = np.where(g_in_blue_row, rb_at_g_col, red)
    red = np.where(mask_b, rb_at_br, red)

    blue = np.where(mask_b, m, 0.0)
    blue = np.where(g_in_blue_row, rb_at_g_row, blue)
    blue = np.where(g_in_red_row, rb_at_g_col, blue)
    blue = np.where(mask_r, rb_at_br, blue)

    return np.stack([red, green, blue], axis=-1).astype(np.float32)


def camera_to_srgb_matrix(rgb_xyz_matrix: Optional[np.ndarray]) -> np.ndarray:
    """Build the camera-RGB -> linear-sRGB matrix, following dcraw.

    libraw hands us ``cam_xyz`` (XYZ -> camera). Composing with sRGB -> XYZ gives
    sRGB -> camera; inverting that gives what we want. The row normalisation is
    dcraw's: it makes a neutral camera signal map to a neutral sRGB one, so the
    matrix does not double up on white balance.
    """
    if rgb_xyz_matrix is None:
        return np.eye(3)
    cam_xyz = np.asarray(rgb_xyz_matrix, dtype=np.float64)[:3, :3]
    if not np.isfinite(cam_xyz).all() or abs(np.linalg.det(cam_xyz)) < 1e-12:
        return np.eye(3)
    cam_rgb = cam_xyz @ XYZ_RGB
    row_sums = cam_rgb.sum(axis=1, keepdims=True)
    row_sums[np.abs(row_sums) < 1e-9] = 1.0
    cam_rgb = cam_rgb / row_sums
    try:
        return np.linalg.inv(cam_rgb)
    except np.linalg.LinAlgError:
        return np.eye(3)


def apply_white_balance(rgb: np.ndarray, multipliers: Sequence[float]) -> np.ndarray:
    """Scale the channels so a neutral subject becomes neutral."""
    wb = np.asarray(multipliers, dtype=np.float32)[:3].copy()
    if not np.isfinite(wb).all() or wb.max() <= 0:
        return rgb
    wb[wb <= 0] = 1.0
    wb = wb / wb[1]  # normalise to green, which is the reference channel
    return rgb * wb.reshape(1, 1, 3)


def srgb_gamma(linear: np.ndarray) -> np.ndarray:
    """The real sRGB transfer function, linear toe included."""
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def develop(
    img: RawImage,
    method: str = "malvar",
    white_balance: bool = True,
    color_matrix: bool = True,
    exposure: float = 0.0,
    gamma: bool = True,
    auto_bright: bool = False,
    auto_bright_percentile: float = 99.5,
) -> np.ndarray:
    """Develop a :class:`RawImage` into an ``(H, W, 3)`` image in ``[0, 1]``.

    ``exposure`` is in stops. ``auto_bright`` rescales so the given percentile
    hits white -- convenient for looking at underexposed low-light frames, but
    it changes the tonality, so it is off by default.
    """
    data = np.clip(img.data, 0.0, None)
    if img.is_bayer:
        rgb = demosaic(data, img.pattern, method=method)
    elif data.ndim == 2:
        rgb = np.repeat(data[..., None], 3, axis=-1)
    else:
        rgb = data

    if white_balance:
        rgb = apply_white_balance(rgb, img.camera_whitebalance)

    if exposure:
        rgb = rgb * (2.0 ** exposure)

    if auto_bright:
        ref = float(np.percentile(rgb, auto_bright_percentile))
        if ref > 1e-6:
            rgb = rgb / ref

    if color_matrix:
        mat = camera_to_srgb_matrix(img.color_matrix)
        rgb = np.einsum("ij,hwj->hwi", mat.astype(np.float32), rgb)

    rgb = np.clip(rgb, 0.0, 1.0)
    return srgb_gamma(rgb).astype(np.float32) if gamma else rgb
