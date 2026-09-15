"""Training loop.

Nothing exotic, but the details that matter for restoration are here:

* **EMA of the weights.** Restoration metrics are noisy from step to step; an
  exponential moving average is consistently worth 0.1-0.2 dB and costs nothing.
* **Cosine schedule with warmup.** The standard recipe for NAFNet/Restormer.
  Warmup matters because the first steps of a deep residual stack are unstable.
* **Gradient clipping**, for the same reason.
* **AMP** on CUDA, which roughly halves memory and lets you train at a patch
  size where the network actually sees structure.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..models.registry import save_checkpoint
from .losses import CombinedLoss, psnr


@dataclass
class TrainConfig:
    preset: str = "nafnet"
    conditioning: str = "sigma_map"
    channels: int = 4
    epochs: int = 100
    batch_size: int = 8
    patch_size: int = 128
    lr: float = 1e-3
    min_lr: float = 1e-6
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    amp: bool = True
    num_workers: int = 4
    seed: int = 0
    out_dir: str = "runs/default"
    log_every: int = 50
    val_every: int = 1
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"charbonnier": 1.0, "psnr": 0.0, "frequency": 0.0, "shadow": 0.0}
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class EMA:
    """Exponential moving average of model parameters, with warmup.

    A fixed decay of 0.999 has a time constant of ~1000 steps. On a short run
    the average never escapes its starting point -- and because residual models
    here are deliberately initialised to the identity, that means the saved
    checkpoint is an identity function that does nothing at all. It looks like a
    model that simply failed to learn.

    The standard fix (used by TensorFlow's ExponentialMovingAverage and timm) is
    to ramp the decay in: early on the average tracks the weights closely, and
    it relaxes to the requested decay once enough steps have accumulated.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, warmup: bool = True):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        decay = self.current_decay()
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(decay).add_(v.detach().float(), alpha=1.0 - decay)

    def copy_to(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Swap EMA weights in, returning the originals so they can be restored."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)
        return backup

    def restore(self, model: torch.nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        model.load_state_dict({**model.state_dict(), **backup}, strict=False)


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(total - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    total, count, base = 0.0, 0, 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        a = batch["a"].to(device)
        b = batch["b"].to(device)
        pred = model(noisy, a, b)
        total += float(psnr(pred.clamp(0, 1), clean.clamp(0, 1)))
        base += float(psnr(noisy.clamp(0, 1), clean.clamp(0, 1)))
        count += 1
    model.train()
    if count == 0:
        return {"psnr": float("nan"), "psnr_input": float("nan"), "gain": float("nan")}
    return {"psnr": total / count, "psnr_input": base / count, "gain": (total - base) / count}


def train(
    model: torch.nn.Module,
    train_set,
    val_set=None,
    cfg: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
    resume: Optional[str] = None,
):
    """Run training and return the path of the best checkpoint."""
    cfg = cfg or TrainConfig()
    from .infer import pick_device

    device = device or pick_device("auto")
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as fh:
        fh.write(cfg.to_json())

    model = model.to(device)
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=cfg.num_workers > 0,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=max(1, cfg.batch_size // 2), shuffle=False,
                                num_workers=min(cfg.num_workers, 2))

    criterion = CombinedLoss(**cfg.loss_weights).to(device)
    # NAFNet trains with betas=(0.9, 0.9). That low beta2 makes Adam very
    # responsive to recent gradients, which needs a carefully tuned learning
    # rate; at a general-purpose default it can leave a residual model
    # oscillating around its identity initialisation and never converging.
    # (0.9, 0.999) is the robust choice; set cfg.betas to reproduce the paper.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  betas=tuple(cfg.betas))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    total_steps = cfg.epochs * max(len(train_loader), 1)
    if total_steps <= cfg.warmup_steps:
        import warnings
        warnings.warn(
            f"this run is {total_steps} steps but warmup alone is {cfg.warmup_steps}, so the "
            "learning rate never reaches its peak and the model will barely move. Lower "
            "--lr warmup, or train for longer.",
            RuntimeWarning, stacklevel=2,
        )
    step = 0
    best_psnr = -float("inf")
    start_epoch = 0
    history: list = []

    if resume and os.path.exists(resume):
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = payload.get("epoch", 0) + 1
        step = payload.get("step", 0)
        best_psnr = payload.get("best_psnr", -float("inf"))
        if ema is not None and payload.get("ema"):
            ema.shadow = {k: v.to(device) for k, v in payload["ema"].items()}
            ema.step = payload.get("ema_step", step)

    log_path = os.path.join(cfg.out_dir, "log.jsonl")
    best_path = os.path.join(cfg.out_dir, "best.ckpt")
    last_path = os.path.join(cfg.out_dir, "last.ckpt")

    def log(record: dict) -> None:
        record["time"] = time.time()
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        running = 0.0
        seen = 0
        for batch in train_loader:
            lr = lr_at(step, total_steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            a = batch["a"].to(device, non_blocking=True)
            b = batch["b"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(noisy, a, b)
                loss, parts = criterion(pred, clean)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            running += float(loss.detach())
            seen += 1
            step += 1
            if step % cfg.log_every == 0:
                log({"epoch": epoch, "step": step, "lr": lr,
                     "loss": running / max(seen, 1), "parts": parts})
                running, seen = 0.0, 0

        metrics = {}
        if val_loader is not None and (epoch + 1) % cfg.val_every == 0:
            backup = ema.copy_to(model) if ema is not None else None
            metrics = evaluate(model, val_loader, device)
            if ema is not None:
                ema.restore(model, backup)
            log({"epoch": epoch, "step": step, "val": metrics})

            history.append(metrics)
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                backup = ema.copy_to(model) if ema is not None else None
                save_checkpoint(best_path, model, epoch=epoch, step=step,
                                best_psnr=best_psnr, metrics=metrics)
                if ema is not None:
                    ema.restore(model, backup)

        save_checkpoint(last_path, model, epoch=epoch, step=step, best_psnr=best_psnr,
                        optimizer=optimizer.state_dict(),
                        ema=(ema.shadow if ema is not None else None),
                        ema_step=(ema.step if ema is not None else 0))

    _warn_if_no_better_than_doing_nothing(history)
    return best_path if os.path.exists(best_path) else last_path


def _warn_if_no_better_than_doing_nothing(history) -> None:
    """Shout if the finished model does not beat leaving the image alone.

    A residual network starts as the identity, so a run that fails to converge
    produces a checkpoint that returns its input unchanged. That is easy to miss
    -- the run exits cleanly, the loss looks small, and the checkpoint loads --
    and the first sign is a denoise that reports 0.0 dB. Checking here turns a
    silent non-result into an explicit one.
    """
    if not history:
        return
    best = max(history, key=lambda m: m["psnr"])
    if not (best["psnr"] > best["psnr_input"] + 0.05):
        warnings.warn(
            f"training finished without beating the noisy input "
            f"({best['psnr']:.2f} dB vs {best['psnr_input']:.2f} dB): this checkpoint is "
            "effectively a no-op. The usual causes are a learning rate too high for the "
            "model size (try --lr 2e-4), too few steps, or too little training data.",
            RuntimeWarning, stacklevel=2,
        )
