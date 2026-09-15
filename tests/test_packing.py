"""CFA packing must be exactly invertible and pattern-correct."""

import numpy as np
import pytest

from denoiseraw.rawio.packing import (
    channel_color_index,
    pack_bayer,
    pad_to_multiple,
    unpack_bayer,
    unpad,
)

PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_pack_unpack_roundtrip(pattern):
    mosaic = np.arange(16 * 20, dtype=np.float32).reshape(16, 20)
    assert np.array_equal(unpack_bayer(pack_bayer(mosaic, pattern), pattern), mosaic)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_packed_planes_are_r_g_g_b(pattern):
    """Planes are always emitted as R, G1, G2, B whatever the sensor's layout."""
    assert list(channel_color_index(pattern)) == [0, 1, 1, 2]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_packed_planes_hold_the_right_colour(pattern):
    """Each packed plane must contain exactly the sites of its own colour."""
    lut = {"R": 0.0, "G": 1.0, "B": 2.0}
    mosaic = np.zeros((8, 8), dtype=np.float32)
    for i, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        mosaic[dy::2, dx::2] = lut[pattern[i]]
    packed = pack_bayer(mosaic, pattern)
    assert np.all(packed[0] == 0.0)          # R plane
    assert np.all(packed[1] == 1.0)          # G1
    assert np.all(packed[2] == 1.0)          # G2
    assert np.all(packed[3] == 2.0)          # B


def test_odd_sizes_are_cropped_to_even():
    mosaic = np.random.rand(9, 11).astype(np.float32)
    assert pack_bayer(mosaic, "RGGB").shape == (4, 4, 5)
    assert unpack_bayer(pack_bayer(mosaic, "RGGB"), "RGGB").shape == (8, 10)


def test_unknown_pattern_rejected():
    with pytest.raises(ValueError, match="unsupported CFA pattern"):
        pack_bayer(np.zeros((4, 4), np.float32), "XTRANS")


def test_padding_roundtrip():
    arr = np.random.rand(4, 7, 13).astype(np.float32)
    padded, pad = pad_to_multiple(arr, 8)
    assert padded.shape == (4, 8, 16)
    assert np.allclose(unpad(padded, pad), arr)
