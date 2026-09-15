"""Demosaic and colour handling."""

import numpy as np
import pytest

from denoiseraw.rawio.develop import (
    DEMOSAIC_METHODS,
    apply_white_balance,
    camera_to_srgb_matrix,
    demosaic,
    develop,
    srgb_gamma,
)

PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]
_LUT = {"R": 0, "G": 1, "B": 2}


def mosaic_from_rgb(rgb, pattern):
    out = np.zeros(rgb.shape[:2], np.float32)
    for letter, (dy, dx) in zip(pattern, ((0, 0), (0, 1), (1, 0), (1, 1))):
        out[dy::2, dx::2] = rgb[dy::2, dx::2, _LUT[letter]]
    return out


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("method", DEMOSAIC_METHODS)
def test_constant_image_is_reproduced_exactly(pattern, method):
    rgb = np.full((32, 32, 3), 0.37, np.float32)
    out = demosaic(mosaic_from_rgb(rgb, pattern), pattern, method)
    assert np.abs(out[4:-4, 4:-4] - 0.37).max() < 1e-5


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("method", DEMOSAIC_METHODS)
def test_linear_ramp_is_reproduced_exactly(pattern, method):
    yy, xx = np.mgrid[0:32, 0:32]
    rgb = np.stack([0.1 + 0.02 * xx, 0.2 + 0.01 * yy, 0.3 + 0.015 * xx], -1).astype(np.float32)
    out = demosaic(mosaic_from_rgb(rgb, pattern), pattern, method)
    assert np.abs(out[4:-4, 4:-4] - rgb[4:-4, 4:-4]).max() < 1e-5


def test_malvar_beats_bilinear_on_real_content():
    """Malvar's whole purpose is gradient correction; if it is not clearly ahead
    of bilinear on a real photograph the kernels or the site masks are wrong."""
    from skimage.data import astronaut
    from skimage.metrics import peak_signal_noise_ratio

    rgb = astronaut().astype(np.float32) / 255.0
    mosaic = mosaic_from_rgb(rgb, "RGGB")
    scores = {m: peak_signal_noise_ratio(rgb[6:-6, 6:-6],
                                         demosaic(mosaic, "RGGB", m)[6:-6, 6:-6], data_range=1)
              for m in ("bilinear", "malvar")}
    assert scores["malvar"] > scores["bilinear"] + 2.0


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="unknown demosaic method"):
        demosaic(np.zeros((8, 8), np.float32), "RGGB", "guesswork")


def test_colour_matrix_keeps_neutral_neutral():
    """dcraw's row normalisation means a neutral camera signal must stay
    neutral, so the matrix does not silently redo white balance."""
    cam_xyz = np.array([[0.6446, -0.0366, -0.0864],
                        [-0.4436, 1.2204, 0.2513],
                        [-0.0952, 0.1873, 0.6607],
                        [0.0, 0.0, 0.0]])
    matrix = camera_to_srgb_matrix(cam_xyz)
    assert np.allclose(matrix @ np.ones(3), np.ones(3), atol=1e-6)


def test_degenerate_colour_matrix_falls_back_to_identity():
    assert np.allclose(camera_to_srgb_matrix(np.zeros((4, 3))), np.eye(3))
    assert np.allclose(camera_to_srgb_matrix(None), np.eye(3))


def test_white_balance_normalises_to_green():
    rgb = np.ones((4, 4, 3), np.float32)
    out = apply_white_balance(rgb, [2.0, 1.0, 1.5, 1.0])
    assert np.allclose(out[0, 0], [2.0, 1.0, 1.5])
    # Degenerate multipliers must not destroy the image.
    assert np.allclose(apply_white_balance(rgb, [0.0, 0.0, 0.0, 0.0]), rgb)


def test_srgb_gamma_endpoints_and_monotonicity():
    x = np.linspace(0, 1, 101)
    y = srgb_gamma(x)
    assert y[0] == pytest.approx(0.0)
    assert y[-1] == pytest.approx(1.0)
    assert np.all(np.diff(y) >= 0)
    assert srgb_gamma(np.array([0.0031308]))[0] == pytest.approx(0.0031308 * 12.92, rel=1e-3)


def test_develop_produces_a_sane_image(synthetic_dng):
    from denoiseraw.rawio.loader import load_raw

    path, _, _ = synthetic_dng
    rgb = develop(load_raw(path))
    assert rgb.shape[2] == 3
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    assert rgb.std() > 0.02          # not a flat grey
