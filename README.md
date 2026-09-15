# DeNoiseRaw

Noise reduction for DSLR and mirrorless RAW files, built on the physics of how
sensors actually make noise.

It works on the mosaic, before demosaicing, and it models the noise rather than
guessing at it: photon shot noise, heavy-tailed read noise, row banding and
quantisation, calibrated to your camera. The result goes back out as a linear
DNG you finish in Lightroom, Capture One or darktable.

```bash
pip install -e .

denoiseraw denoise IMG_1234.CR2 -o IMG_1234_denoised.dng
```

That works immediately, with no model weights and no GPU. Train a network when
you want the last few dB.

---

## Why not just use the denoiser in my raw converter?

Most of them denoise *after* demosaicing, often after the tone curve. By then
the noise is spatially correlated (demosaicing mixes neighbours), channel
correlated (the colour matrix mixes channels) and non-linearly transformed — so
there is no longer a model that describes it, and the denoiser is reduced to
generic smoothing.

In RAW, before any of that, noise is spatially independent and follows a
two-parameter law you can measure:

```
Var(x) = a·x + b
```

`a` is photon shot noise, `b` is everything signal-independent. Knowing those
two numbers turns denoising from guesswork into estimation — the denoiser is
told exactly how uncertain every pixel is, instead of inferring it from texture
and getting it wrong on both smooth skies and fine foliage.

---

## What it does

**Physics-based noise model.** The full ELD formulation (Wei et al., CVPR 2020):
Poisson shot noise, Tukey-lambda read noise, per-row banding and quantisation.
The heavy tails and the banding are what separate a denoiser that survives deep
shadows from one that does not.

**Noise parameters from your camera, three ways.**
1. *Calibrated* — shoot lens-cap frames and a flat-field ramp once; exact.
2. *Blind* — fitted from the photograph itself, Foi et al. (TIP 2008). Within a
   few percent on ordinary scenes.
3. *Given* — pass a profile JSON.

**Networks.** Faithful implementations of NAFNet (ECCV 2022) and Restormer
(CVPR 2022), plus a separable-convolution U-Net for CPU work. Parameter counts
are asserted against the published configurations in the test suite, so they
cannot silently drift from the papers.

**Per-pixel noise conditioning.** The network receives `sqrt(a·x + b)` as extra
input planes, so one model covers every ISO instead of one model per ISO.

**Works without training.** BM3D under a variance-stabilising transform, with
the closed-form unbiased inverse (Mäkitalo & Foi, TIP 2011).

**Adapts to your sensor without clean data.** `denoiseraw finetune` uses
Neighbor2Neighbor (CVPR 2021) to fine-tune on your own noisy files.

**Linear DNG output.** Pixel-exact, opens in any raw converter, keeps your
processing decisions for later.

Full reasoning and citations: [`docs/PAPERS.md`](docs/PAPERS.md).

---

## Usage

### Denoise

```bash
# One file. Method 'auto' uses a model if you point it at weights, else classical.
denoiseraw denoise IMG_1234.CR2 -o out.dng

# A whole shoot
denoiseraw denoise ~/shoot/*.NEF -o ~/shoot/denoised/

# With a trained model, tiled for a 45 MP file, plus 8x self-ensembling
denoiseraw denoise IMG.ARW -o out.dng \
    --checkpoint runs/nafnet/best.ckpt --tile 512 --self-ensemble

# Gentler: keep more grain and fine detail
denoiseraw denoise IMG.CR3 -o out.dng --strength 0.6

# A viewable preview instead of a DNG
denoiseraw denoise IMG.CR2 -o preview.jpg --auto-bright
```

`--strength` works by over- or understating sigma to the conditioned model, so
it is a real change in how much noise the model thinks is there, not a blend
with the original.

### Inspect

```bash
denoiseraw info IMG_1234.CR2        # camera, CFA, black/white levels, ISO
denoiseraw estimate IMG_1234.CR2    # blind noise estimate for this frame
```

### Calibrate your camera (once per body)

Worth ten minutes. Measured parameters do not care what the scene looks like,
which is exactly where blind estimation is weakest.

```bash
# Bias frames: lens cap on, fastest shutter, at the ISO you care about.
# Flat frames: an evenly lit blank wall, defocused, bracketed from near-black
#              to near-clipping, TWO frames at each level.
denoiseraw calibrate \
    --bias bias/*.CR2 \
    --flat flat/*.CR2 \
    --iso 3200 \
    -o profiles/5d4_iso3200.json

denoiseraw denoise IMG.CR2 -o out.dng --profile profiles/5d4_iso3200.json
```

Bias frames alone give you the read-noise floor, the banding level and the
read-noise shape. Flats add the shot-noise slope.

### Train

```bash
# Point it at clean, low-ISO, well-exposed RAW files. Noise is synthesised
# on the fly from the calibrated model, resampled every epoch.
denoiseraw train --data ~/raw/clean/ --preset nafnet \
    --profile profiles/5d4_iso3200.json \
    --epochs 200 --batch-size 16 --patch-size 256 \
    --out-dir runs/nafnet

# Real paired data (SIDD, or your own tripod pairs)
denoiseraw train --paired-noisy sidd/noisy/ --paired-clean sidd/clean/ \
    --preset nafnet --out-dir runs/nafnet-ft
```

Presets: `lite`, `nafnet-small`, `nafnet`, `nafnet-large`, `restormer-small`,
`restormer`. Ready-made recipes are in [`configs/`](configs/).

### Fine-tune to your camera, no ground truth needed

```bash
denoiseraw finetune ~/shoot/*.CR2 \
    --checkpoint runs/nafnet/best.ckpt \
    --steps 2000 -o my_5d4.ckpt
```

### Compare methods on your own file

```bash
denoiseraw bench IMG.CR2 --checkpoint runs/nafnet/best.ckpt
```

Re-noises a crop of your own image with a known profile, so PSNR, SSIM and a
detail-preservation ratio are all measurable against real ground truth.

---

## Python API

```python
from denoiseraw import DenoiseSettings, denoise_file, write_outputs

result = denoise_file("IMG_1234.CR2", DenoiseSettings(
    method="model",
    checkpoint="runs/nafnet/best.ckpt",
    self_ensemble=True,
))

print(result.profile)                       # the noise model that was used
print(result.metrics["noise_reduction_db"])
write_outputs(result, "out.dng")
```

---

## How it is put together

```
denoiseraw/
├── rawio/          RAW in, developed or linear DNG out
│   ├── loader.py       libraw -> normalised linear RawImage
│   ├── packing.py      Bayer mosaic <-> 4 colour-consistent planes
│   ├── develop.py      Malvar demosaic, white balance, colour matrix, sRGB
│   └── writer.py       DNG / linear CFA TIFF / preview images
├── noise/          the scientific core
│   ├── profile.py      Var = a·x + b, per-ISO banks
│   ├── synth.py        ELD noise synthesis for training
│   ├── estimate.py     blind estimation from one frame
│   └── calibrate.py    bias / flat-field calibration
├── models/         NAFNet, Restormer, a lite U-Net, and noise conditioning
├── engine/         tiled inference, training, Neighbor2Neighbor, D4 geometry
├── classical/      BM3D under a variance-stabilising transform
├── data/           synthetic and paired datasets
├── vst.py          generalised Anscombe transform (NumPy and Torch)
├── metrics.py      PSNR, SSIM, detail preservation, residual noise
└── pipeline.py     the whole thing, end to end
```

Two details that are easy to get wrong and are handled carefully:

**Row noise must follow physical sensor rows.** In packed Bayer, the R and G1
planes come from the *same* sensor row and must share the banding offset; G2 and
B come from the next one. Getting this wrong turns coherent banding into
incoherent per-plane noise, and the model never learns to remove the real thing.

**Flipping packed Bayer permutes the colour planes.** Mirroring an RGGB mosaic
horizontally makes it read GRBG, so in packed terms R and G1 swap planes on top
of the spatial flip. Ignore that and self-ensembling averages eight mutually
inconsistent estimates, quietly making the output *worse*. `engine/geometry.py`
handles it, and `tests/test_geometry.py` checks it against the mosaic ground
truth.

---

## Measured behaviour

From the test suite, on synthetic captures with a known profile:

| Property | Result |
|---|---|
| Noise synthesis | reproduces `Var = a·x + b` to within 6% at every level, all three models |
| Calibration (bias + flats) | recovers `a` to 0.5%, `b` to 6%, banding to 4% |
| Blind estimation | sigma at 10% grey within ~15%, biased slightly high (safe direction) |
| VST | holds noise sigma at 1.00 across three decades of signal |
| Unbiased GAT inverse | ~15× less bias than the algebraic inverse |
| Malvar demosaic | +3.8 dB over bilinear on real content; exact on constant and linear ramps |
| Tiled inference | no step discontinuity at tile boundaries for overlap ≥ 16 |
| DNG export | pixel-exact round trip through libraw; colour matrix honoured |
| Training | a small model gains +2.6 dB over the noisy input in a 3-minute CPU run |
| Training (`lite`, high ISO) | +6.1 dB over a 29.3 dB input in 30 epochs on CPU |

Reference points from the literature, on the SIDD benchmark (sRGB), for a sense
of what trained models are worth:

| Model | Params | SIDD PSNR |
|---|---:|---:|
| BM3D (no training) | – | 25.65 dB |
| MIRNet | 31.8 M | 39.72 dB |
| Uformer-B | 50.9 M | 39.89 dB |
| Restormer | 26.1 M | 40.02 dB |
| NAFNet-width64 | 116 M | 40.30 dB |

### Tested against real camera files

Genuine CR2/NEF captures (Canon EOS 5D Mark II at ISO 3200 f/1.2, Nikon D3S at
ISO 3200 f/1.4 — sourced from rawpy's public test fixtures, not synthetic), run
through the classical (BM3D-under-VST / wavelet) path with no trained weights:

| File | Backend | Time | Noise reduction* |
|---|---|---:|---:|
| 5D Mark II, 5634×3752 | wavelet | 5.9 s | 33.7 dB |
| 5D Mark II, 5634×3752 | bm3d | 5m18s | 31.7 dB |
| D3S, 4284×2844 | wavelet | 3.7 s | 26.5 dB |

\*Measured on matched flat-scene blocks selected from the *noisy* input only
(`paired_noise_reduction`), not independently from each image — see the note
below on why that distinction matters. No ground truth exists for a real
photograph, so this is the best available proxy, not a PSNR.

**A number this large deserved scrutiny before being written down.** An
earlier version of this metric selected each image's "flattest" blocks
*independently*; on a real photo that lets an over-smoothed, texture-destroyed
region in the output masquerade as evidence of noise removal it never
performed elsewhere. Verifying it required checking three separate things: an
independent high-pass measurement on hand-picked patches (agreed, ~99% noise
variance removed even in a textured road patch — the earlier concern that this
was "too good" turned out to conflate real low-frequency scene structure with
actual per-pixel noise, which a naive raw-patch standard deviation cannot tell
apart); a constructed adversarial case that *does* fool the independent-
selection method (65.7 dB reported for zero real improvement) to confirm the
failure mode is real; and only then confirming that this particular photo's
number holds up under the corrected, pairwise measurement (33.65 dB vs the
original 33.6 dB — it wasn't, in fact, what was fooling the metric here). Both
the fix and the adversarial regression test are in `tests/test_pipeline.py`.
What the large dB figure does *not* establish is whether fine real texture
(gravel-scale detail in the road, for instance) survived alongside the noise —
that needs a clean reference this photo doesn't have, so treat it as unverified
rather than assume it's fine.

**No pre-trained weights ship with this repository.** The classical path runs
out of the box; the network path needs you to train or supply a checkpoint, and
`--method model` fails loudly rather than quietly returning an untrained
network's output.

---

## Honest limitations

- **Canon sRAW/mRAW modes are not supported and crash.** These modes have the
  camera do partial demosaicing in hardware, so `raw_image_visible` comes back
  as an already-multi-channel `(H, W, N)` array instead of a 2-D mosaic, which
  nothing downstream expects. Found by testing against a real Canon 40D sRAW
  file; full-resolution RAW (CR2/CR3/NEF/ARW/...) from the same bodies works
  normally. Shoot full-resolution RAW if your camera offers a choice.
- **X-Trans and Quad-Bayer** sensors fall back to single-plane processing.
  Correct, but it forfeits the colour-consistency advantage of packing, so
  expect less from Fujifilm files.
- **Blind noise estimation degrades on frames textured at fine scale
  everywhere** — dense foliage, fabric, gravel filling the frame. A high-pass
  filter cannot distinguish near-Nyquist detail from noise, so the estimate
  biases towards over-smoothing. Calibrate if this matters to you.
- **The bundled BM3D is 0.7–1.1 dB behind the reference `bm3d` package** and
  roughly ten times slower, because the reference is compiled and uses a
  bi-orthogonal wavelet in its first pass. Install `bm3d` and it will be
  preferred automatically; the measurements are in the module docstring.
- **EXIF is not carried into the output DNG.** The sensor data, CFA pattern,
  black and white levels, white balance and colour matrix all survive; shooting
  metadata does not. Copy it across with `exiftool` if you need it.
- **Values below 0 ADU cannot be stored** in an unsigned RAW container. This
  never arises on a real capture, where the black level leaves ample headroom,
  but the writers warn rather than clipping silently.
- **`--strength` above ~2 will visibly plasticise skin and foliage.** It is a
  real change to the assumed noise level, not a cosmetic blend.
- **Short training runs on nearly-clean data can end up as no-ops.** A residual
  network starts as the identity, and if there is little noise to remove and few
  steps to learn from, it stays there — the checkpoint loads fine and then
  reports 0.0 dB. `denoiseraw train` now warns explicitly when a finished run
  never beat its own noisy input, but the fix is more data, more steps, or
  training at the ISO you actually shoot.

---

## Development

```bash
pip install -e ".[dev]"
pytest                       # ~190 tests, about 30 seconds
pytest -m slow               # plus a real (short) training run
```

The test suite manufactures synthetic camera files — a scene with realistic
frequency content, mosaicked, corrupted with the physics model, written as a
DNG that libraw opens exactly like a camera's own file — so everything is
tested end to end without shipping a 30 MB sample.

## Licence

MIT.
