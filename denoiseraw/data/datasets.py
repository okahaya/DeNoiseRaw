"""Training data for RAW denoising.

Two regimes are supported.

**Synthetic pairs from clean RAW files** (the default). Point the trainer at a
folder of low-ISO, well-exposed RAW files; each sample crops a patch and injects
noise drawn from a calibrated profile. This is how *Unprocessing* (CVPR 2019) and
ELD (CVPR 2020) produce training data, and it is the only practical way to cover
a continuum of ISOs.

**Real paired data** (SIDD, or your own tripod-shot pairs): directories of
matching noisy/clean files. More faithful, far more limited in coverage. The
usual recipe -- and the one the config files here follow -- is to pre-train on
synthetic and fine-tune on real.

Patches are cached to a single ``.npy`` stack on first use, because decoding a
RAW file takes ~0.5 s and would otherwise dominate every epoch.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from ..noise.profile import NoiseProfile, ProfileBank
from ..noise.synth import sample_profile, synthesize_noise
from ..rawio.packing import pack_bayer

RAW_EXTENSIONS = (".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".dng",
                  ".orf", ".pef", ".srw", ".raw", ".3fr", ".iiq")


def list_images(root: str) -> List[str]:
    """All RAW / cached-array files under ``root``, sorted for reproducibility."""
    out = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(RAW_EXTENSIONS + (".npy",)):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def load_packed(path: str) -> np.ndarray:
    """Load a file as ``(4, H, W)`` packed planes in normalised units."""
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        if arr.ndim == 2:
            raise ValueError(f"{path}: 2-D .npy needs a CFA pattern; pre-pack it instead")
        return arr.astype(np.float32)
    from ..rawio.loader import load_raw

    img = load_raw(path)
    if not img.is_bayer:
        return img.data[None].astype(np.float32)
    return pack_bayer(img.data, img.pattern)


@dataclass
class PatchCache:
    """A cached stack of clean patches, keyed by the source list and settings."""

    path: str
    patches: np.ndarray

    @staticmethod
    def build(sources: Sequence[str], patch_size: int, per_image: int,
              cache_dir: str, seed: int = 0, min_brightness: float = 0.01) -> "PatchCache":
        key = hashlib.sha1(
            ("|".join(sources) + f"|{patch_size}|{per_image}|{seed}|{min_brightness}").encode()
        ).hexdigest()[:16]
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"patches_{key}.npy")
        if os.path.exists(path):
            return PatchCache(path, np.load(path, mmap_mode="r"))

        rng = np.random.default_rng(seed)
        collected = []
        for src in sources:
            try:
                packed = load_packed(src)
            except (OSError, ValueError, RuntimeError):
                # A corrupt or unsupported file in a big library should not
                # abort a training run; skip it and carry on.
                continue
            _, h, w = packed.shape
            if h < patch_size or w < patch_size:
                continue
            tries = 0
            taken = 0
            while taken < per_image and tries < per_image * 8:
                tries += 1
                y = int(rng.integers(0, h - patch_size + 1))
                x = int(rng.integers(0, w - patch_size + 1))
                patch = packed[:, y:y + patch_size, x:x + patch_size]
                # Skip near-black patches: they teach the model nothing except
                # to output zero, and they are abundant in most RAW libraries.
                if float(patch.mean()) < min_brightness:
                    continue
                collected.append(np.ascontiguousarray(patch, dtype=np.float32))
                taken += 1
        if not collected:
            raise RuntimeError("no usable patches found; check the source directory")
        stack = np.stack(collected)
        np.save(path, stack)
        return PatchCache(path, np.load(path, mmap_mode="r"))


class SyntheticNoiseDataset(Dataset):
    """Clean patches plus freshly synthesised noise on every access.

    Re-sampling the noise each epoch (rather than fixing it once) is a big win:
    the model sees the same scene under many noise realisations and cannot
    memorise a particular one.
    """

    def __init__(
        self,
        sources: Sequence[str],
        patch_size: int = 128,
        per_image: int = 32,
        cache_dir: str = ".cache/denoiseraw",
        profile: Optional[NoiseProfile] = None,
        bank: Optional[ProfileBank] = None,
        iso_range: Tuple[float, float] = (100.0, 25600.0),
        noise_model: str = "eld",
        augment: bool = True,
        length: Optional[int] = None,
        seed: int = 0,
    ):
        if profile is None and bank is None:
            raise ValueError("provide either a NoiseProfile or a ProfileBank")
        self.cache = PatchCache.build(list(sources), patch_size, per_image, cache_dir, seed)
        self.profile = profile
        self.bank = bank
        self.iso_range = iso_range
        self.noise_model = noise_model
        self.augment = augment
        self.length = length or len(self.cache.patches)
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        import torch

        from ..engine.geometry import apply_transform, transform_sequences

        # A per-sample RNG keeps workers independent and the run reproducible.
        rng = np.random.default_rng((self.seed * 1_000_003 + index) % (2 ** 63))
        patches = self.cache.patches
        clean = np.array(patches[int(rng.integers(len(patches)))], dtype=np.float32)

        if self.augment:
            seqs = transform_sequences()
            seq = seqs[int(rng.integers(len(seqs)))]
            # CFA-aware: a flip permutes the packed planes as well as the pixels.
            clean = np.ascontiguousarray(apply_transform(clean, seq, packed=clean.shape[0] == 4))

        prof = sample_profile(self.bank or self.profile, self.iso_range, rng=rng)
        noisy = synthesize_noise(clean, prof, model=self.noise_model, packed=True, rng=rng)

        a, b = prof.per_channel_arrays(clean.shape[0])
        return {
            "noisy": torch.from_numpy(noisy),
            "clean": torch.from_numpy(clean),
            "a": torch.from_numpy(a.astype(np.float32)),
            "b": torch.from_numpy(b.astype(np.float32)),
        }


class PairedDataset(Dataset):
    """Real (noisy, clean) pairs, matched by filename."""

    def __init__(
        self,
        noisy_dir: str,
        clean_dir: str,
        patch_size: int = 128,
        augment: bool = True,
        profile: Optional[NoiseProfile] = None,
        estimate_noise: bool = True,
        length: Optional[int] = None,
        seed: int = 0,
    ):
        noisy_files = {os.path.basename(p): p for p in list_images(noisy_dir)}
        clean_files = {os.path.basename(p): p for p in list_images(clean_dir)}
        shared = sorted(set(noisy_files) & set(clean_files))
        if not shared:
            raise RuntimeError(f"no matching filenames between {noisy_dir} and {clean_dir}")
        self.pairs = [(noisy_files[n], clean_files[n]) for n in shared]
        self.patch_size = patch_size
        self.augment = augment
        self.profile = profile
        self.estimate_noise = estimate_noise
        self.length = length or len(self.pairs)
        self.seed = seed
        self._cache: dict = {}

    def __len__(self) -> int:
        return self.length

    def _get(self, i: int):
        if i not in self._cache:
            if len(self._cache) > 8:          # bounded: full RAWs are large
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = (load_packed(self.pairs[i][0]), load_packed(self.pairs[i][1]))
        return self._cache[i]

    def __getitem__(self, index: int):
        import torch

        from ..engine.geometry import apply_transform, transform_sequences
        from ..noise.estimate import estimate_profile

        rng = np.random.default_rng((self.seed * 7_919 + index) % (2 ** 63))
        noisy_full, clean_full = self._get(int(rng.integers(len(self.pairs))))

        c, h, w = noisy_full.shape
        ps = min(self.patch_size, h, w)
        y = int(rng.integers(0, h - ps + 1))
        x = int(rng.integers(0, w - ps + 1))
        noisy = np.array(noisy_full[:, y:y + ps, x:x + ps], dtype=np.float32)
        clean = np.array(clean_full[:, y:y + ps, x:x + ps], dtype=np.float32)

        if self.augment:
            seqs = transform_sequences()
            seq = seqs[int(rng.integers(len(seqs)))]
            packed = c == 4
            noisy = np.ascontiguousarray(apply_transform(noisy, seq, packed=packed))
            clean = np.ascontiguousarray(apply_transform(clean, seq, packed=packed))

        prof = self.profile
        if prof is None and self.estimate_noise:
            # Estimating from the residual is exact here -- we have the clean
            # reference -- and far more reliable than a blind single-image fit.
            prof = _profile_from_residual(noisy, clean)
        elif prof is None:
            prof = estimate_profile(noisy)

        a, b = prof.per_channel_arrays(c)
        return {
            "noisy": torch.from_numpy(noisy),
            "clean": torch.from_numpy(clean),
            "a": torch.from_numpy(a.astype(np.float32)),
            "b": torch.from_numpy(b.astype(np.float32)),
        }


def _profile_from_residual(noisy: np.ndarray, clean: np.ndarray) -> NoiseProfile:
    """Fit ``Var = a x + b`` to the known residual of a real pair."""
    resid = (noisy - clean).ravel()
    level = clean.ravel()
    order = np.argsort(level)
    level, resid = level[order], resid[order]
    n_bins = 16
    edges = np.linspace(0, len(level), n_bins + 1, dtype=int)
    xs, ys = [], []
    for i in range(n_bins):
        seg = resid[edges[i]:edges[i + 1]]
        if len(seg) < 64:
            continue
        xs.append(float(level[edges[i]:edges[i + 1]].mean()))
        ys.append(float(seg.var()))
    if len(xs) < 2:
        return NoiseProfile(a=0.0, b=float(np.var(resid)) or 1e-8)
    a, b = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    return NoiseProfile(a=float(max(a, 0.0)), b=float(max(b, 1e-12)), source="paired-residual")
