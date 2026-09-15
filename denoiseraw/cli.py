"""Command-line interface.

    denoiseraw denoise  IMG.CR2 -o out.dng       # the main event
    denoiseraw info     IMG.CR2                  # what's in this file?
    denoiseraw estimate IMG.CR2                  # blind noise estimate
    denoiseraw calibrate --bias ... --flat ...   # measure your sensor properly
    denoiseraw train    --data clean_raws/       # train a model
    denoiseraw finetune IMG.CR2 --checkpoint ... # adapt to your camera, no GT
    denoiseraw bench    IMG.CR2                  # compare methods on one file
    denoiseraw gui                               # point-and-click web interface
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional, Sequence

import numpy as np


def _expand(patterns: Sequence[str]) -> List[str]:
    """Expand globs and directories into a sorted file list."""
    out: List[str] = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            from .data.datasets import list_images
            out.extend(list_images(pattern))
        elif any(ch in pattern for ch in "*?["):
            out.extend(sorted(glob.glob(pattern)))
        else:
            out.append(pattern)
    return out


# --------------------------------------------------------------------------
# denoise
# --------------------------------------------------------------------------
def cmd_denoise(args) -> int:
    from .pipeline import DenoiseSettings, denoise_file, write_outputs

    files = _expand(args.input)
    if not files:
        print("no input files matched", file=sys.stderr)
        return 1

    settings = DenoiseSettings(
        method=args.method, checkpoint=args.checkpoint, strength=args.strength,
        classical_backend=args.backend, profile_path=args.profile, iso=args.iso,
        tile=args.tile, overlap=args.overlap, self_ensemble=args.self_ensemble,
        device=args.device, demosaic=args.demosaic, exposure=args.exposure,
        auto_bright=args.auto_bright, progress=not args.quiet,
        extra={"linear_cfa": args.linear_cfa},
    )

    multiple = len(files) > 1
    if multiple and args.output and not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    failures = 0
    for path in files:
        try:
            result = denoise_file(path, settings)
        except Exception as exc:                      # keep going through a batch
            print(f"{os.path.basename(path)}: FAILED - {exc}", file=sys.stderr)
            failures += 1
            continue

        if multiple or (args.output and os.path.isdir(args.output)):
            stem = os.path.splitext(os.path.basename(path))[0]
            out_dir = args.output or os.path.dirname(path) or "."
            output = os.path.join(out_dir, f"{stem}_denoised{args.suffix}")
        else:
            output = args.output or os.path.splitext(path)[0] + f"_denoised{args.suffix}"

        written = write_outputs(result, output, settings)
        if not args.quiet:
            m = result.metrics
            print(f"{os.path.basename(path)}  [{result.method}]  {result.elapsed:.1f}s  "
                  f"noise {m['residual_noise_before']:.5f} -> {m['residual_noise_after']:.5f} "
                  f"({m.get('noise_reduction_db', 0):.1f} dB)  -> {next(iter(written.values()))}")
    return 1 if failures else 0


# --------------------------------------------------------------------------
# info / estimate
# --------------------------------------------------------------------------
def cmd_info(args) -> int:
    from .rawio.loader import load_raw

    for path in _expand(args.input):
        img = load_raw(path)
        print(f"\n{path}")
        print(f"  camera        : {img.camera_model}")
        print(f"  sensor        : {img.shape[1]} x {img.shape[0]}  CFA={img.pattern or 'non-Bayer'}")
        print(f"  black / white : {img.black_level:.1f} / {img.white_level:.1f}  "
              f"({img.scale:.0f} ADU span)")
        print(f"  ISO / shutter : {img.iso or '?'} / {img.exposure_time or '?'}s  "
              f"f/{img.aperture or '?'}")
        print(f"  camera WB     : {np.round(img.camera_whitebalance, 3).tolist()}")
        print(f"  data range    : {img.data.min():.4f} .. {img.data.max():.4f} (normalised)")
    return 0


def cmd_estimate(args) -> int:
    from .noise.estimate import estimate_profile
    from .rawio.loader import load_raw

    for path in _expand(args.input):
        img = load_raw(path)
        prof = estimate_profile(img)
        print(f"\n{path}")
        print(f"  {prof}")
        print(f"  sigma at 1% / 10% / 50% grey: {prof.sigma_at(0.01):.5f} / "
              f"{prof.sigma_at(0.10):.5f} / {prof.sigma_at(0.50):.5f}")
        if args.output:
            prof.save(args.output)
            print(f"  written to {args.output}")
    return 0


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------
def cmd_calibrate(args) -> int:
    from .noise.calibrate import calibrate

    bias = _expand(args.bias)
    flats = _expand(args.flat) if args.flat else []
    if not bias:
        print("at least one bias (lens-cap) frame is required", file=sys.stderr)
        return 1

    prof = calibrate(bias, flats, iso=args.iso)
    print(prof)
    if not flats:
        print("  note: without flat frames the shot-noise slope 'a' stays 0. "
              "Shoot an exposure ramp of a blank wall, two frames per level.")
    prof.save(args.output)
    print(f"  written to {args.output}")
    return 0


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------
def cmd_train(args) -> int:
    from .data import PairedDataset, SyntheticNoiseDataset, list_images
    from .engine.train import TrainConfig, train
    from .models import build_model
    from .noise.profile import NoiseProfile, ProfileBank

    cfg = TrainConfig(
        preset=args.preset, epochs=args.epochs, batch_size=args.batch_size,
        patch_size=args.patch_size, lr=args.lr, num_workers=args.workers,
        out_dir=args.out_dir, amp=not args.no_amp, conditioning=args.conditioning,
    )
    if args.loss:
        cfg.loss_weights = json.loads(args.loss)

    if args.paired_noisy and args.paired_clean:
        train_set = PairedDataset(args.paired_noisy, args.paired_clean,
                                  patch_size=args.patch_size)
        val_set = None
    else:
        sources = list_images(args.data)
        if not sources:
            print(f"no RAW files found under {args.data}", file=sys.stderr)
            return 1
        split = max(1, int(len(sources) * 0.9))
        profile = bank = None
        if args.profile:
            try:
                bank = ProfileBank.load(args.profile)
                if not bank.profiles:
                    bank = None
            except (KeyError, ValueError, TypeError):
                bank = None
            if bank is None:
                profile = NoiseProfile.load(args.profile)
        else:
            # A generic modern CMOS profile: fine for pre-training, but calibrate
            # your own body before you trust the result on real files.
            profile = NoiseProfile(a=3e-4, b=1.5e-5, sigma_row=1e-3,
                                   tukey_lambda=0.14, quantization_step=1 / 16383)
        common = dict(patch_size=args.patch_size, per_image=args.patches_per_image,
                      cache_dir=args.cache_dir, profile=profile, bank=bank,
                      iso_range=(args.iso_min, args.iso_max), noise_model=args.noise_model)
        train_set = SyntheticNoiseDataset(sources[:split], seed=1, **common)
        val_set = SyntheticNoiseDataset(sources[split:] or sources[:2], seed=99,
                                        augment=False, **common)

    model = build_model(args.preset, conditioning=args.conditioning)
    from .models import parameter_count
    print(f"training {args.preset} ({parameter_count(model)/1e6:.2f}M params), "
          f"conditioning={args.conditioning}")
    best = train(model, train_set, val_set, cfg, resume=args.resume)
    print(f"best checkpoint: {best}")
    return 0


# --------------------------------------------------------------------------
# finetune (self-supervised)
# --------------------------------------------------------------------------
def cmd_finetune(args) -> int:
    import torch

    from .engine.infer import pick_device
    from .engine.selfsup import neighbor2neighbor_loss
    from .models import build_model, load_checkpoint, save_checkpoint
    from .noise.estimate import estimate_profile
    from .rawio.loader import load_raw
    from .rawio.packing import pack_bayer

    files = _expand(args.input)
    if not files:
        print("no input files matched", file=sys.stderr)
        return 1

    device = pick_device(args.device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        model, _ = load_checkpoint(args.checkpoint)
    else:
        model = build_model(args.preset)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Cache packed planes and their profiles once; decoding RAW every step would
    # dominate the runtime.
    samples = []
    for path in files:
        img = load_raw(path)
        packed = pack_bayer(img.data, img.pattern) if img.is_bayer else img.data[None]
        samples.append((packed, estimate_profile(img)))
    print(f"fine-tuning on {len(samples)} file(s), {args.steps} steps, device={device}")

    rng = np.random.default_rng(0)
    ps = args.patch_size
    for step in range(args.steps):
        batch, a_list, b_list = [], [], []
        for _ in range(args.batch_size):
            packed, prof = samples[int(rng.integers(len(samples)))]
            c, h, w = packed.shape
            y = int(rng.integers(0, max(h - ps, 1)))
            x = int(rng.integers(0, max(w - ps, 1)))
            batch.append(packed[:, y:y + ps, x:x + ps])
            a, b = prof.per_channel_arrays(c)
            a_list.append(a)
            b_list.append(b)

        noisy = torch.from_numpy(np.stack(batch)).to(device)
        a = torch.from_numpy(np.stack(a_list).astype(np.float32)).to(device)
        b = torch.from_numpy(np.stack(b_list).astype(np.float32)).to(device)

        optimizer.zero_grad(set_to_none=True)
        loss, parts = neighbor2neighbor_loss(model, noisy, a, b, gamma=args.gamma)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % max(args.steps // 20, 1) == 0:
            print(f"  step {step:5d}  loss={float(loss.detach()):.6f}  "
                  f"rec={parts['reconstruction']:.6f} reg={parts['regularisation']:.6f}")

    save_checkpoint(args.output, model, finetuned_on=[os.path.basename(f) for f in files])
    print(f"written to {args.output}")
    return 0


# --------------------------------------------------------------------------
# bench
# --------------------------------------------------------------------------
def cmd_bench(args) -> int:
    """Compare methods on one file by re-noising it with a known profile."""
    from .metrics import evaluate_pair
    from .noise.profile import NoiseProfile
    from .noise.synth import synthesize_noise
    from .pipeline import DenoiseSettings, denoise_packed_array
    from .rawio.loader import load_raw
    from .rawio.packing import pack_bayer

    img = load_raw(_expand(args.input)[0])
    clean = pack_bayer(img.data, img.pattern) if img.is_bayer else img.data[None]
    if args.crop:
        clean = clean[:, : args.crop, : args.crop]

    prof = NoiseProfile(a=args.a, b=args.b, sigma_row=args.row,
                        tukey_lambda=0.14, quantization_step=1 / img.scale)
    noisy = synthesize_noise(clean, prof, model="eld", rng=np.random.default_rng(0))

    print(f"{'method':<22} {'PSNR':>8} {'SSIM':>7} {'detail':>7}")
    base = evaluate_pair(noisy, clean)
    print(f"{'(noisy input)':<22} {base['psnr']:8.2f} {base['ssim']:7.3f} "
          f"{base['detail_preservation']:7.3f}")

    methods = [("classical", dict(method="classical", classical_backend=b))
               for b in args.backends.split(",")]
    if args.checkpoint:
        methods.append(("model", dict(method="model", checkpoint=args.checkpoint)))
        methods.append(("model+ensemble", dict(method="model", checkpoint=args.checkpoint,
                                               self_ensemble=True)))
    for name, kwargs in methods:
        label = f"{name}:{kwargs.get('classical_backend', '')}".rstrip(":")
        try:
            out = denoise_packed_array(noisy, prof, DenoiseSettings(progress=False, **kwargs))
        except Exception as exc:
            print(f"{label:<22} FAILED: {exc}")
            continue
        m = evaluate_pair(out, clean)
        print(f"{label:<22} {m['psnr']:8.2f} {m['ssim']:7.3f} {m['detail_preservation']:7.3f}")
    return 0


# --------------------------------------------------------------------------
# gui
# --------------------------------------------------------------------------
def cmd_gui(args) -> int:
    try:
        from .gui import build_interface
    except ImportError:
        print("The GUI needs gradio. Install it with `pip install -e \".[gui]\"` "
              "or `pip install gradio`.", file=sys.stderr)
        return 1

    demo = build_interface()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="denoiseraw", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("denoise", help="denoise one or more RAW files")
    d.add_argument("input", nargs="+", help="RAW files, globs or directories")
    d.add_argument("-o", "--output", help="output file, or directory for a batch")
    d.add_argument("--suffix", default=".dng",
                   help="output extension when -o is omitted (.dng/.tif/.png/.jpg)")
    d.add_argument("--method", default="auto", choices=["auto", "model", "classical", "none"])
    d.add_argument("--checkpoint", help="trained model weights")
    d.add_argument("--backend", default="auto",
                   help="classical backend: auto/bm3d/bm3d-lite/wavelet/nlm")
    d.add_argument("--strength", type=float, default=1.0,
                   help="<1 keeps more grain and detail, >1 smooths harder")
    d.add_argument("--profile", help="calibrated noise profile JSON")
    d.add_argument("--iso", type=float, help="override the ISO used to pick a profile")
    d.add_argument("--tile", type=int, default=512)
    d.add_argument("--overlap", type=int, default=64)
    d.add_argument("--self-ensemble", action="store_true",
                   help="average over 8 flips/rotations: slower, slightly better")
    d.add_argument("--device", default="auto")
    d.add_argument("--demosaic", default="malvar", choices=["malvar", "bilinear"])
    d.add_argument("--exposure", type=float, default=0.0, help="preview exposure, in stops")
    d.add_argument("--auto-bright", action="store_true")
    d.add_argument("--linear-cfa", action="store_true",
                   help="write a linear CFA TIFF + JSON sidecar instead of a preview")
    d.add_argument("-q", "--quiet", action="store_true")
    d.set_defaults(func=cmd_denoise)

    i = sub.add_parser("info", help="print RAW metadata")
    i.add_argument("input", nargs="+")
    i.set_defaults(func=cmd_info)

    e = sub.add_parser("estimate", help="estimate noise parameters from an image")
    e.add_argument("input", nargs="+")
    e.add_argument("-o", "--output", help="write the profile as JSON")
    e.set_defaults(func=cmd_estimate)

    c = sub.add_parser("calibrate", help="calibrate from bias and flat frames")
    c.add_argument("--bias", nargs="+", required=True, help="lens-cap frames")
    c.add_argument("--flat", nargs="*", help="exposure ramp, two frames per level")
    c.add_argument("--iso", type=float)
    c.add_argument("-o", "--output", default="noise_profile.json")
    c.set_defaults(func=cmd_calibrate)

    t = sub.add_parser("train", help="train a denoiser")
    t.add_argument("--data", default="data/clean", help="directory of clean low-ISO RAW files")
    t.add_argument("--paired-noisy", help="directory of noisy frames (real pairs)")
    t.add_argument("--paired-clean", help="matching clean frames")
    t.add_argument("--preset", default="nafnet")
    t.add_argument("--conditioning", default="sigma_map",
                   choices=["sigma_map", "vst", "none"])
    t.add_argument("--profile", help="calibrated noise profile or bank JSON")
    t.add_argument("--noise-model", default="eld", choices=["eld", "poisson_gaussian", "gaussian"])
    t.add_argument("--iso-min", type=float, default=100.0)
    t.add_argument("--iso-max", type=float, default=25600.0)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--patch-size", type=int, default=128)
    t.add_argument("--patches-per-image", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--cache-dir", default=".cache/denoiseraw")
    t.add_argument("--out-dir", default="runs/default")
    t.add_argument("--loss", help='JSON weights, e.g. \'{"charbonnier":1.0,"shadow":0.5}\'')
    t.add_argument("--resume")
    t.add_argument("--no-amp", action="store_true")
    t.set_defaults(func=cmd_train)

    f = sub.add_parser("finetune", help="self-supervised fine-tune on your own noisy RAWs")
    f.add_argument("input", nargs="+")
    f.add_argument("--checkpoint", help="starting weights (optional)")
    f.add_argument("--preset", default="lite")
    f.add_argument("-o", "--output", default="finetuned.ckpt")
    f.add_argument("--steps", type=int, default=2000)
    f.add_argument("--batch-size", type=int, default=4)
    f.add_argument("--patch-size", type=int, default=128)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--gamma", type=float, default=2.0,
                   help="Neighbor2Neighbor regularisation weight")
    f.add_argument("--device", default="auto")
    f.set_defaults(func=cmd_finetune)

    b = sub.add_parser("bench", help="compare methods on a file with synthetic noise")
    b.add_argument("input", nargs=1)
    b.add_argument("--checkpoint")
    b.add_argument("--backends", default="bm3d,wavelet")
    b.add_argument("--crop", type=int, default=512)
    b.add_argument("--a", type=float, default=3e-4, help="shot-noise slope")
    b.add_argument("--b", type=float, default=1.6e-5, help="read-noise variance")
    b.add_argument("--row", type=float, default=1e-3, help="row-noise sigma")
    b.set_defaults(func=cmd_bench)

    g = sub.add_parser("gui", help="launch the point-and-click web interface")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=None)
    g.add_argument("--share", action="store_true", help="create a public share link")
    g.set_defaults(func=cmd_gui)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
