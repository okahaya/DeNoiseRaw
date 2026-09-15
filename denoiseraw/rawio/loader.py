"""Loading DSLR/mirrorless RAW files into a normalised, linear representation.

The rest of the package never touches libraw directly; it works on :class:`RawImage`,
which carries the sensor data in *linear normalised* form (0 = black level,
1 = saturation) together with the metadata needed to (a) synthesise physically
correct noise and (b) develop the result back into a viewable image.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

# Canonical CFA layouts we understand natively. Anything else (X-Trans, Quad-Bayer)
# is handled in single-channel mode, see :attr:`RawImage.is_bayer`.
BAYER_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")

_LIBRAW_COLOR_LETTERS = "RGBG"  # libraw colour indices 0..3


@dataclass
class RawImage:
    """A RAW capture in linear, black-subtracted, normalised space.

    Attributes
    ----------
    data:
        ``float32`` array of shape ``(H, W)`` holding the visible sensor area.
        Values are ``(adu - black_level) / (white_level - black_level)``; they are
        *not* clipped, so genuine below-black noise survives (important: clipping
        the left tail biases dark regions green/magenta).
    pattern:
        One of :data:`BAYER_PATTERNS`, or ``None`` for non-Bayer sensors.
    black_level / white_level:
        Per-CFA-channel levels in ADU, as reported by libraw, in the order
        implied by ``pattern``.
    camera_whitebalance:
        As-shot RGBG multipliers.
    color_matrix:
        3x4 camera-RGB -> XYZ(D65) matrix from libraw (``rgb_xyz_matrix``).
    """

    data: np.ndarray
    pattern: Optional[str]
    black_level: float
    white_level: float
    black_level_per_channel: Sequence[float] = field(default_factory=list)
    camera_whitebalance: Sequence[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    daylight_whitebalance: Sequence[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    color_matrix: Optional[np.ndarray] = None
    camera_model: str = "unknown"
    iso: Optional[float] = None
    exposure_time: Optional[float] = None
    aperture: Optional[float] = None
    source_path: Optional[str] = None
    # libraw's colour index per position of the 2x2 (or NxN) CFA tile
    raw_pattern: Optional[np.ndarray] = None

    # -- basic properties -------------------------------------------------
    @property
    def is_bayer(self) -> bool:
        return self.pattern in BAYER_PATTERNS

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def scale(self) -> float:
        """ADU span of the normalised [0, 1] range."""
        return float(self.white_level - self.black_level)

    def to_adu(self, normalised: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert normalised data back to raw ADU."""
        arr = self.data if normalised is None else normalised
        return arr * self.scale + self.black_level

    def replaced(self, data: np.ndarray) -> "RawImage":
        """Return a copy carrying ``data`` instead of the current pixels."""
        clone = RawImage(**{**asdict(self), "data": data})
        clone.color_matrix = self.color_matrix
        clone.raw_pattern = self.raw_pattern
        return clone

    def metadata(self) -> dict:
        return {
            "pattern": self.pattern,
            "black_level": self.black_level,
            "white_level": self.white_level,
            "black_level_per_channel": list(map(float, self.black_level_per_channel)),
            "camera_whitebalance": list(map(float, self.camera_whitebalance)),
            "daylight_whitebalance": list(map(float, self.daylight_whitebalance)),
            "color_matrix": None if self.color_matrix is None else self.color_matrix.tolist(),
            "camera_model": self.camera_model,
            "iso": self.iso,
            "exposure_time": self.exposure_time,
            "aperture": self.aperture,
            "source_path": self.source_path,
        }


def _pattern_from_raw(raw) -> Optional[str]:
    """Derive an ``RGGB``-style string from libraw's ``raw_pattern``."""
    try:
        rp = np.asarray(raw.raw_pattern)
    except Exception:
        return None
    if rp is None or rp.shape != (2, 2):
        return None
    letters = "".join(_LIBRAW_COLOR_LETTERS[int(i)] for i in rp.reshape(-1))
    return letters if letters in BAYER_PATTERNS else None


def load_raw(path: str, use_camera_wb: bool = True) -> RawImage:
    """Read a RAW file (CR2/CR3/NEF/ARW/RAF/RW2/DNG/...) into a :class:`RawImage`."""
    import rawpy

    with rawpy.imread(path) as raw:
        visible = np.asarray(raw.raw_image_visible, dtype=np.float32)
        pattern = _pattern_from_raw(raw)

        black_per_channel = list(np.asarray(raw.black_level_per_channel, dtype=np.float64))
        # libraw exposes a per-pixel black map for some sensors; prefer the scalar mean
        # for normalisation and keep the per-channel values for accurate development.
        black = float(np.mean(black_per_channel)) if black_per_channel else 0.0
        white = float(raw.white_level)
        if raw.camera_white_level_per_channel is not None:
            wl = [w for w in raw.camera_white_level_per_channel if w]
            if wl:
                white = float(np.min(wl))
        if white <= black:  # defensive: some bodies report nonsense
            white = black + float(np.percentile(visible, 99.99) - black) or black + 1.0

        scale = max(white - black, 1e-6)
        data = (visible - black) / scale

        cm = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)

        img = RawImage(
            data=data.astype(np.float32),
            pattern=pattern,
            black_level=black,
            white_level=white,
            black_level_per_channel=black_per_channel,
            camera_whitebalance=list(np.asarray(raw.camera_whitebalance, dtype=np.float64)),
            daylight_whitebalance=list(np.asarray(raw.daylight_whitebalance, dtype=np.float64)),
            color_matrix=cm,
            source_path=os.path.abspath(path),
            raw_pattern=np.asarray(raw.raw_pattern) if pattern else None,
        )

    _attach_exif(img, path)
    if not use_camera_wb:
        img.camera_whitebalance = list(img.daylight_whitebalance)
    return img


def _attach_exif(img: RawImage, path: str) -> None:
    """Best-effort EXIF enrichment (camera model / ISO / shutter / aperture)."""
    try:
        import exifread
    except ImportError:
        return
    try:
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
    except Exception:
        return

    def _num(key):
        tag = tags.get(key)
        if tag is None:
            return None
        try:
            values = tag.values
            v = values[0] if isinstance(values, (list, tuple)) else values
            return float(getattr(v, "num", v)) / float(getattr(v, "den", 1))
        except Exception:
            return None

    make = str(tags.get("Image Make", "")).strip()
    model = str(tags.get("Image Model", "")).strip()
    if make or model:
        # Bodies usually repeat the brand in the model field ("NIKON" + "NIKON Z 6").
        img.camera_model = model if model.upper().startswith(make.upper()) else f"{make} {model}".strip()
    img.iso = _num("EXIF ISOSpeedRatings")
    img.exposure_time = _num("EXIF ExposureTime")
    img.aperture = _num("EXIF FNumber")


def save_sidecar(img: RawImage, path: str) -> None:
    """Write the metadata needed to re-develop a denoised CFA TIFF."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(img.metadata(), fh, indent=2)


def load_sidecar(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
