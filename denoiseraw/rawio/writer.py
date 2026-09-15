"""Writing results out.

Three kinds of output, for three different intentions:

``save_image``
    A developed 8- or 16-bit RGB file (PNG / TIFF / JPEG) for looking at and
    sharing.

``save_linear_cfa``
    The denoised mosaic itself, as a 16-bit linear TIFF plus a JSON sidecar with
    the black/white levels, white balance and colour matrix. This is the
    important one for real work: it keeps the data linear and undeveloped so you
    can finish the photo in your own raw converter.

``save_dng``
    The same thing as an actual DNG, which raw converters open directly. Written
    with tifffile using the DNG tag set.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Optional

import numpy as np

from .loader import RawImage
from .packing import cfa_color_index


def _quantise_adu(adu: np.ndarray, what: str) -> np.ndarray:
    """Encode sensor ADU as unsigned 16-bit, warning about anything clipped.

    RAW containers store unsigned integers, so a value below 0 ADU cannot be
    represented. On a real capture that never comes up: the black level (a few
    hundred ADU) leaves ample room for the below-black noise tail, which the
    rest of this package is careful to preserve. But a denoised or synthetic
    frame can undershoot, and losing that tail biases the shadows -- so say so
    rather than clipping quietly.
    """
    below = int(np.count_nonzero(adu < 0.0))
    above = int(np.count_nonzero(adu > 65535.0))
    if below or above:
        total = adu.size
        warnings.warn(
            f"{what}: {below} pixel(s) below 0 ADU and {above} above 65535 were clipped "
            f"({100.0 * (below + above) / total:.3f}% of the frame). Unsigned RAW containers "
            "cannot hold sub-zero values; if this is a large fraction, the black level is "
            "probably wrong for this data.",
            RuntimeWarning, stacklevel=3,
        )
    return np.clip(np.rint(adu), 0, 65535).astype(np.uint16)


def _to_uint(arr: np.ndarray, bits: int) -> np.ndarray:
    maxv = (1 << bits) - 1
    return np.clip(np.rint(np.clip(arr, 0.0, 1.0) * maxv), 0, maxv).astype(
        np.uint16 if bits > 8 else np.uint8
    )


def save_image(path: str, rgb: np.ndarray, bits: Optional[int] = None,
               quality: int = 95) -> str:
    """Write a developed ``(H, W, 3)`` image in ``[0, 1]``.

    ``bits`` defaults to the best depth the format supports. Note that PNG *the
    format* allows 16-bit RGB but Pillow cannot write it, so 16-bit RGB output
    has to go to TIFF; asking for it as PNG raises rather than silently
    discarding 8 bits of the shadow detail this package works to preserve.
    """
    ext = os.path.splitext(path)[1].lower()
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    is_rgb = rgb.ndim == 3 and rgb.shape[-1] == 3

    if ext in (".jpg", ".jpeg"):
        if bits not in (None, 8):
            raise ValueError("JPEG is 8-bit only; use .tif for more depth")
        import imageio.v3 as iio
        iio.imwrite(path, _to_uint(rgb, 8), quality=quality)

    elif ext in (".tif", ".tiff"):
        import tifffile
        tifffile.imwrite(path, _to_uint(rgb, bits or 16), photometric="rgb")

    elif ext == ".png":
        depth = bits if bits is not None else (8 if is_rgb else 16)
        if depth > 8 and is_rgb:
            raise ValueError(
                "16-bit RGB PNG cannot be written by Pillow; use a .tif output "
                "for 16-bit, or pass bits=8 to accept an 8-bit PNG"
            )
        import imageio.v3 as iio
        iio.imwrite(path, _to_uint(rgb, depth))

    else:
        raise ValueError(f"unsupported output extension {ext!r}; use .png, .tif or .jpg")
    return path


def save_linear_cfa(path: str, img: RawImage, sidecar: bool = True) -> str:
    """Write the (denoised) mosaic as a 16-bit linear TIFF plus metadata.

    The data is re-encoded to the original ADU scale, so black and white levels
    in the sidecar remain meaningful and the file is a drop-in replacement for
    the sensor data of the original capture.
    """
    import tifffile

    adu = img.to_adu()
    white = max(img.white_level, 1.0)
    tifffile.imwrite(path, _quantise_adu(adu, "linear CFA TIFF"), photometric="minisblack")

    if sidecar:
        meta = img.metadata()
        meta["encoding"] = {
            "description": "16-bit linear CFA, values in original sensor ADU",
            "white_level": white,
            "black_level": img.black_level,
            "cfa_pattern": img.pattern,
        }
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    return path


def save_dng(path: str, img: RawImage, description: str = "Denoised with DeNoiseRaw") -> str:
    """Write the denoised mosaic as a DNG that raw converters can open.

    DNG is a constrained TIFF, so tifffile can produce one given the right tags.
    The essential set is: the DNG version, a CFA photometric interpretation with
    its pattern, the black and white levels, the as-shot neutral (white balance)
    and a colour matrix tying the camera's primaries to XYZ.
    """
    import tifffile

    if not img.is_bayer:
        raise ValueError("DNG export currently supports Bayer sensors only")

    adu = _quantise_adu(img.to_adu(), "DNG")
    h, w = adu.shape
    adu = adu[: h - h % 2, : w - w % 2]

    cfa = cfa_color_index(img.pattern).astype(np.uint8).reshape(-1)

    # As-shot neutral is the reciprocal of the white-balance multipliers.
    wb = np.asarray(img.camera_whitebalance, dtype=np.float64)[:3]
    wb = np.where(np.isfinite(wb) & (wb > 0), wb, 1.0)
    wb = wb / wb[1]
    as_shot_neutral = (1.0 / wb).astype(np.float64)

    cm = img.color_matrix
    colour_matrix = (np.asarray(cm, dtype=np.float64)[:3, :3]
                     if cm is not None else np.eye(3))

    def srational(values):
        """DNG wants SRATIONAL pairs; 10000 is plenty of precision here."""
        out = []
        for v in np.asarray(values, dtype=np.float64).reshape(-1):
            out.extend([round(v * 10000), 10000])
        return out

    make, _, model = (img.camera_model or "unknown").partition(" ")
    extratags = [
        (271, "s", 0, make or "unknown", True),                  # Make
        (272, "s", 0, model or img.camera_model or "unknown", True),  # Model
        (50706, "B", 4, (1, 4, 0, 0), True),                     # DNGVersion 1.4
        (50707, "B", 4, (1, 1, 0, 0), True),                     # DNGBackwardVersion
        (33421, "H", 2, (2, 2), True),                           # CFARepeatPatternDim
        (33422, "B", 4, tuple(int(v) for v in cfa), True),       # CFAPattern
        (50708, "s", 0, img.camera_model or "unknown", True),    # UniqueCameraModel
        (50717, "I", 1, (int(img.white_level),), True),          # WhiteLevel
        (50714, "I", 1, (round(img.black_level),), True),        # BlackLevel
        (50721, "2i", 9, tuple(srational(colour_matrix)), True),  # ColorMatrix1
        (50728, "2i", 3, tuple(srational(as_shot_neutral)), True),  # AsShotNeutral
        (50778, "H", 1, (21,), True),                            # CalibrationIlluminant1 = D65
    ]

    # Note: libraw fills ``rgb_xyz_matrix`` from its own camera database rather
    # than from ColorMatrix1, so reading this file back with rawpy shows zeros
    # there even though the tag is present and correct. The matrix *is* used
    # during development -- see tests/test_writer.py, which verifies that
    # changing it changes libraw's rendered output.

    tifffile.imwrite(
        path,
        adu,
        photometric="cfa",
        planarconfig="contig",
        compression=None,
        description=description,
        software="DeNoiseRaw",
        extratags=extratags,
    )
    return path
