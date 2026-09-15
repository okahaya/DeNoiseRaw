"""Output files must be readable by the tools people actually use."""

import json

import numpy as np
import pytest

from denoiseraw.rawio.loader import RawImage, load_raw
from denoiseraw.rawio.writer import save_dng, save_image, save_linear_cfa

CAM_XYZ = np.array([[0.6446, -0.0366, -0.0864],
                    [-0.4436, 1.2204, 0.2513],
                    [-0.0952, 0.1873, 0.6607],
                    [0.0, 0.0, 0.0]])


def make_image(h=64, w=80):
    yy, xx = np.mgrid[0:h, 0:w]
    # Stays above -black/scale, i.e. within the black-level headroom a real
    # sensor leaves for its below-black noise tail.
    data = (0.12 + 0.55 * (xx / w) + 0.1 * np.sin(yy / 9.0)).astype(np.float32)
    return RawImage(data=data, pattern="RGGB", black_level=512.0, white_level=16383.0,
                    black_level_per_channel=[512.0] * 4,
                    camera_whitebalance=[2.1, 1.0, 1.5, 1.0],
                    color_matrix=CAM_XYZ, camera_model="Canon EOS 5D Mark IV")


def test_dng_is_readable_by_libraw_and_pixel_exact(tmp_path):
    img = make_image()
    path = str(tmp_path / "out.dng")
    save_dng(path, img)

    import rawpy
    with rawpy.imread(path) as raw:
        back = np.asarray(raw.raw_image_visible)
        expected = np.clip(np.rint(img.to_adu()), 0, 65535).astype(np.uint16)
        assert np.array_equal(back, expected)
        assert list(raw.black_level_per_channel)[:1] == [512]
        assert raw.white_level == 16383
        assert np.allclose(np.asarray(raw.camera_whitebalance)[:3], [2.1, 1.0, 1.5], atol=1e-3)


def test_dng_roundtrips_through_our_own_loader(tmp_path):
    img = make_image()
    path = str(tmp_path / "out.dng")
    save_dng(path, img)
    back = load_raw(path)
    assert back.pattern == "RGGB"
    assert back.white_level == 16383
    assert np.abs(back.data - img.data).max() < 2.0 / img.scale   # 16-bit quantisation


def test_dng_colour_matrix_is_honoured_when_rendering(tmp_path):
    """libraw exposes rgb_xyz_matrix from its own camera database rather than
    from ColorMatrix1, so check the tag is *used* instead of merely present."""
    import rawpy

    renders = []
    for matrix in (CAM_XYZ, np.array([[0.9, -0.3, -0.1], [-0.2, 1.0, 0.2],
                                      [-0.05, 0.1, 0.8], [0.0, 0.0, 0.0]])):
        img = make_image()
        img.color_matrix = matrix
        path = str(tmp_path / f"m{len(renders)}.dng")
        save_dng(path, img)
        with rawpy.imread(path) as raw:
            renders.append(raw.postprocess(use_camera_wb=True, output_bps=16).astype(np.float64))
    assert np.abs(renders[0] - renders[1]).mean() > 50


def test_dng_rejects_non_bayer(tmp_path):
    img = make_image()
    img.pattern = None
    with pytest.raises(ValueError, match="Bayer"):
        save_dng(str(tmp_path / "x.dng"), img)


def test_linear_cfa_writes_adu_and_sidecar(tmp_path):
    import tifffile

    img = make_image()
    path = str(tmp_path / "linear.tif")
    save_linear_cfa(path, img)

    data = tifffile.imread(path)
    assert data.dtype == np.uint16
    assert np.abs(data.astype(np.float64) - img.to_adu()).max() <= 0.5

    with open(str(tmp_path / "linear.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["pattern"] == "RGGB"
    assert meta["white_level"] == 16383.0
    assert meta["encoding"]["cfa_pattern"] == "RGGB"


@pytest.mark.parametrize("name,expected_dtype", [
    ("a.png", np.uint8), ("a.tif", np.uint16), ("a.jpg", np.uint8)])
def test_image_formats(tmp_path, name, expected_dtype):
    import imageio.v3 as iio

    rgb = np.random.rand(32, 32, 3).astype(np.float32)
    path = str(tmp_path / name)
    save_image(path, rgb)
    assert iio.imread(path).dtype == expected_dtype


def test_refuses_depths_the_format_cannot_hold(tmp_path):
    rgb = np.random.rand(16, 16, 3).astype(np.float32)
    with pytest.raises(ValueError, match="16-bit RGB PNG"):
        save_image(str(tmp_path / "x.png"), rgb, bits=16)
    with pytest.raises(ValueError, match="8-bit only"):
        save_image(str(tmp_path / "x.jpg"), rgb, bits=16)
    with pytest.raises(ValueError, match="unsupported output extension"):
        save_image(str(tmp_path / "x.bmp"), rgb)


def test_writers_warn_instead_of_clipping_silently(tmp_path):
    """Sub-zero ADU cannot be stored in an unsigned container. Losing that tail
    biases the shadows, so it must never happen quietly."""
    img = make_image()
    img.data = img.data - 0.2          # drive part of the frame below 0 ADU
    with pytest.warns(RuntimeWarning, match="below 0 ADU"):
        save_dng(str(tmp_path / "clipped.dng"), img)
    with pytest.warns(RuntimeWarning, match="below 0 ADU"):
        save_linear_cfa(str(tmp_path / "clipped.tif"), img)


def test_writers_are_silent_for_representable_data(tmp_path, recwarn):
    save_dng(str(tmp_path / "ok.dng"), make_image())
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
