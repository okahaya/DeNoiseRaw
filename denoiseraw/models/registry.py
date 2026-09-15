"""Model construction from names / config dicts, and checkpoint round-tripping."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .lite import LiteUNet
from .nafnet import NAFNet
from .restormer import Restormer
from .wrapper import DenoiserWrapper, extra_input_channels

# Ready-made configurations. "arch" picks the backbone, the rest are its kwargs.
PRESETS: Dict[str, Dict[str, Any]] = {
    # ~1.5M params, CPU-friendly. The one to reach for without a GPU.
    "lite": {"arch": "lite", "width": 32, "depth": 4, "blocks_per_level": 2},
    # ~7M params. A good laptop-GPU default.
    "nafnet-small": {"arch": "nafnet", "width": 16, "enc_blocks": (2, 2, 4, 8),
                     "middle_blocks": 12, "dec_blocks": (2, 2, 2, 2)},
    # ~29M params, matches the published NAFNet-width32 (39.97 dB on SIDD sRGB).
    "nafnet": {"arch": "nafnet", "width": 32, "enc_blocks": (2, 2, 4, 8),
               "middle_blocks": 12, "dec_blocks": (2, 2, 2, 2)},
    # ~116M params. Best quality, wants a 16 GB+ GPU to train.
    "nafnet-large": {"arch": "nafnet", "width": 64, "enc_blocks": (2, 2, 4, 8),
                     "middle_blocks": 12, "dec_blocks": (2, 2, 2, 2)},
    # ~26M params, matches the published Restormer denoising config.
    "restormer": {"arch": "restormer", "dim": 48, "num_blocks": (4, 6, 6, 8),
                  "num_refinement_blocks": 4, "heads": (1, 2, 4, 8)},
    "restormer-small": {"arch": "restormer", "dim": 24, "num_blocks": (2, 3, 3, 4),
                        "num_refinement_blocks": 2, "heads": (1, 2, 4, 8)},
}

_BACKBONES = {"nafnet": NAFNet, "restormer": Restormer, "lite": LiteUNet}


def build_model(
    preset: str = "nafnet",
    channels: int = 4,
    conditioning: str = "sigma_map",
    residual: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
) -> DenoiserWrapper:
    """Build a conditioned denoiser.

    ``channels`` is 4 for packed Bayer and 1 for non-Bayer sensors handled in
    single-plane mode.
    """
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; available: {sorted(PRESETS)}")
    cfg = dict(PRESETS[preset])
    cfg.update(overrides or {})
    arch = cfg.pop("arch")

    in_channels = channels + extra_input_channels(conditioning, channels)
    backbone = _BACKBONES[arch](
        in_channels=in_channels, out_channels=channels, residual=residual, **cfg
    )
    model = DenoiserWrapper(backbone, conditioning=conditioning, channels=channels)
    model.config = {  # type: ignore[attr-defined]
        "preset": preset, "channels": channels, "conditioning": conditioning,
        "residual": residual, "overrides": overrides or {},
    }
    return model


def save_checkpoint(path: str, model: nn.Module, **extra) -> None:
    """Persist weights *and* the config needed to rebuild the architecture."""
    payload = {
        "state_dict": model.state_dict(),
        "config": getattr(model, "config", {}),
        "format_version": 1,
    }
    payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu", strict: bool = True):
    """Rebuild a model from a checkpoint written by :func:`save_checkpoint`.

    Returns ``(model, payload)`` so callers can also read the training metadata
    (the noise profile the model was trained for, in particular).
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = payload.get("config", {})
    model = build_model(
        preset=cfg.get("preset", "nafnet"),
        channels=cfg.get("channels", 4),
        conditioning=cfg.get("conditioning", "sigma_map"),
        residual=cfg.get("residual", True),
        overrides=cfg.get("overrides", {}),
    )
    model.load_state_dict(payload["state_dict"], strict=strict)
    model.eval()
    return model, payload


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
