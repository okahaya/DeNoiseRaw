"""Camera noise profiles.

Everything here lives in *normalised* units: the signal ``x`` runs from 0 at the
black level to 1 at saturation, and all variances are expressed in ``x`` units.
That keeps profiles comparable across 12-, 14- and 16-bit bodies.

The canonical heteroscedastic (Poisson-Gaussian) relation is

.. math::  \\operatorname{Var}(x) = a\\,x + b

where ``a = K / S`` (``K`` the sensor gain in ADU per electron, ``S`` the ADU
span between black and white level) captures photon shot noise, and
``b = sigma_read^2 / S^2`` captures every signal-independent term.

The ELD model of Wei et al. (CVPR 2020 / TPAMI 2021) refines the ``b`` term into
a heavy-tailed read component (Tukey lambda), a banding component that is
constant along a sensor row, and the quantiser's uniform noise. In deep shadows
those three behave very differently from a Gaussian, which is exactly where
Gaussian-trained denoisers fall apart on real DSLR files.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class NoiseProfile:
    """Noise parameters for one camera at one gain setting.

    Parameters
    ----------
    a:
        Shot-noise slope, ``K / S``. Doubling ISO roughly doubles ``a``.
    b:
        Signal-independent variance floor, ``sigma_read^2 / S^2``.
    sigma_row:
        Standard deviation of the per-row banding offset (normalised units).
        Zero disables the term.
    tukey_lambda:
        Shape of the read-noise distribution. ``lambda = 0`` is logistic
        (heavier tails than Gaussian), ``lambda ~ 0.14`` is approximately
        Gaussian, larger values are lighter-tailed. Most CMOS bodies land in
        ``[-0.3, 0.3]``; negative values mean the pronounced tails that produce
        the isolated hot/cold pixels seen in long exposures.
    quantization_step:
        Width of one ADU in normalised units (``1 / S``). Set to 0 to disable.
    """

    a: float
    b: float
    sigma_row: float = 0.0
    tukey_lambda: float = 0.14
    quantization_step: float = 0.0
    # Optional per-plane (R, G1, G2, B) overrides; empty means "same for all".
    a_per_channel: Sequence[float] = field(default_factory=list)
    b_per_channel: Sequence[float] = field(default_factory=list)
    # Provenance
    camera_model: str = "unknown"
    iso: Optional[float] = None
    scale: Optional[float] = None  # ADU span this profile was derived from
    source: str = "manual"

    # -- derived quantities ----------------------------------------------
    def variance(self, x: np.ndarray) -> np.ndarray:
        """Total per-pixel variance at signal level ``x`` (normalised units)."""
        return np.maximum(self.a, 0.0) * np.maximum(x, 0.0) + max(self.b, 0.0)

    def sigma(self, x: np.ndarray) -> np.ndarray:
        """Per-pixel standard deviation — the map fed to the network."""
        return np.sqrt(self.variance(x))

    def sigma_at(self, x: float = 0.0) -> float:
        return float(math.sqrt(max(self.a, 0.0) * max(x, 0.0) + max(self.b, 0.0)))

    @property
    def read_sigma(self) -> float:
        """Standard deviation of the signal-independent part."""
        return math.sqrt(max(self.b, 0.0))

    def scaled(self, factor: float) -> "NoiseProfile":
        """Profile for the same sensor after multiplying the signal by ``factor``.

        Used when a frame is brightened before denoising (the low-light
        ``ratio`` trick from *Learning to See in the Dark*): the shot term scales
        linearly, the read term quadratically.
        """
        out = NoiseProfile(**asdict(self))
        out.a = self.a * factor
        out.b = self.b * factor * factor
        out.sigma_row = self.sigma_row * factor
        out.quantization_step = self.quantization_step * factor
        out.a_per_channel = [v * factor for v in self.a_per_channel]
        out.b_per_channel = [v * factor * factor for v in self.b_per_channel]
        return out

    def per_channel_arrays(self, n: int = 4):
        """Broadcastable ``(a, b)`` arrays of shape ``(n, 1, 1)``."""
        a = np.asarray(self.a_per_channel if len(self.a_per_channel) == n else [self.a] * n,
                       dtype=np.float32).reshape(n, 1, 1)
        b = np.asarray(self.b_per_channel if len(self.b_per_channel) == n else [self.b] * n,
                       dtype=np.float32).reshape(n, 1, 1)
        return a, b

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["a_per_channel"] = list(map(float, self.a_per_channel))
        d["b_per_channel"] = list(map(float, self.b_per_channel))
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseProfile":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "NoiseProfile":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (f"NoiseProfile(a={self.a:.3e}, b={self.b:.3e}, "
                f"read_sigma={self.read_sigma:.3e}, row={self.sigma_row:.3e}, "
                f"lambda={self.tukey_lambda:.2f}, camera={self.camera_model!r}, iso={self.iso})")


@dataclass
class ProfileBank:
    """A camera's profiles across ISO, plus the log-linear laws between them.

    Calibrating every ISO is tedious, so we fit ``log sigma = alpha * log a + beta``
    across whatever gains *were* measured and interpolate the rest. This is the
    sampling scheme ELD uses to synthesise training noise over a continuum of
    gains rather than the handful that were calibrated.
    """

    camera_model: str = "unknown"
    profiles: Dict[str, NoiseProfile] = field(default_factory=dict)  # keyed by ISO string

    def add(self, profile: NoiseProfile) -> None:
        key = str(int(profile.iso)) if profile.iso else "base"
        self.profiles[key] = profile

    def isos(self) -> np.ndarray:
        return np.array(sorted(float(k) for k in self.profiles if k != "base"))

    def at_iso(self, iso: float) -> NoiseProfile:
        """Nearest-in-log-gain profile, linearly rescaled to the requested ISO.

        Sensor gain is proportional to ISO over the analogue range, so ``a``
        scales with the ISO ratio and ``b`` with its square for the photon-limited
        part. This is an approximation — dual-gain sensors have a discontinuity —
        but it beats using a 100-ISO profile on a 6400-ISO frame.
        """
        if not self.profiles:
            raise ValueError("profile bank is empty")
        isos = self.isos()
        if len(isos) == 0:
            return self.profiles["base"]
        idx = int(np.argmin(np.abs(np.log(isos) - math.log(max(iso, 1.0)))))
        base = self.profiles[str(int(isos[idx]))]
        if base.iso and abs(base.iso - iso) > 1e-6:
            out = base.scaled(iso / base.iso)
            out.iso = iso
            out.camera_model = base.camera_model or self.camera_model
            out.source = f"interpolated from ISO {base.iso:g}"
            return out
        return base

    def fit_log_linear(self) -> Dict[str, float]:
        """Fit ``log b = alpha * log a + beta`` over the calibrated gains."""
        a = np.array([p.a for p in self.profiles.values()])
        b = np.array([p.b for p in self.profiles.values()])
        ok = (a > 0) & (b > 0)
        if ok.sum() < 2:
            return {}
        alpha, beta = np.polyfit(np.log(a[ok]), np.log(b[ok]), 1)
        return {"alpha": float(alpha), "beta": float(beta)}

    def save(self, path: str) -> None:
        payload = {
            "camera_model": self.camera_model,
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
            "log_linear": self.fit_log_linear(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProfileBank":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        bank = cls(camera_model=payload.get("camera_model", "unknown"))
        for k, v in payload.get("profiles", {}).items():
            bank.profiles[k] = NoiseProfile.from_dict(v)
        return bank


def tukey_lambda_std(lam: float) -> float:
    """Standard deviation of the standard Tukey-lambda distribution.

    Needed to normalise samples so that a requested ``sigma`` is the actual
    standard deviation rather than an arbitrary scale parameter.
    """
    from scipy.special import gammaln

    if abs(lam) < 1e-6:
        return math.pi / math.sqrt(3.0)  # logistic
    if lam <= -0.5:
        # Variance is undefined; fall back to a heavy-tailed but finite proxy.
        lam = -0.49
    term = 1.0 / (2.0 * lam + 1.0) - math.exp(2.0 * gammaln(lam + 1.0) - gammaln(2.0 * lam + 2.0))
    var = 2.0 / (lam * lam) * term
    return math.sqrt(max(var, 1e-12))
