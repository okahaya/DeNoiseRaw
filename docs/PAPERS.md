# What this is built on

A short guide to the literature behind each design decision, and why the
alternatives were not chosen. Papers are grouped by the problem they solve.

---

## 1. Why denoise in RAW at all

Denoising *after* the raw converter is fighting the pipeline. Demosaicing
correlates neighbouring pixels, the colour matrix mixes channels, and the tone
curve is strongly non-linear — so by the time you see an 8-bit JPEG, the noise
is spatially correlated, channel-correlated and signal-dependent in a way no
model describes.

In RAW, none of that has happened yet. Noise is spatially independent and obeys
a two-parameter law. Every serious result in the field exploits this.

- **Unprocessing Images for Learned Raw Denoising** — Brooks, Mildenhall, Xue,
  Barron, Chen, Sharlet (CVPR 2019). Shows that inverting the ISP to synthesise
  RAW training data beats training on sRGB, and quantifies the gap.
- **Learning to See in the Dark** — Chen, Chen, Xu, Koltun (CVPR 2018). The
  paper that established the packed-Bayer, end-to-end RAW pipeline used here.
- **CycleISP: Real Image Restoration via Improved Data Synthesis** — Zamir et al.
  (CVPR 2020). Learns the RAW↔sRGB mapping in both directions to generate
  realistic training pairs.

**Consequence in this codebase:** everything happens in normalised linear
sensor space, in `denoiseraw/rawio/packing.py`'s four-plane representation.
Development to sRGB (`rawio/develop.py`) exists only so you can *look* at the
result; the recommended output is a linear DNG you finish in your own converter.

---

## 2. The noise model

The textbook model is heteroscedastic ("Poisson–Gaussian"):

```
Var(x) = a·x + b
```

`a` is photon shot noise (Poisson, so variance equals the mean electron count);
`b` collects everything signal-independent. This is accurate in normal light and
is what the classical path and the `sigma_map` conditioning use.

It is *not* enough in deep shadows, which is where high-ISO RAW denoising is
judged. There, three things the Gaussian model misses dominate:

- **Read noise is heavy-tailed**, not Gaussian — hence the isolated bright/dark
  pixels in long exposures.
- **Banding**: readout amplifiers are shared along a sensor row, so their
  fluctuation is one offset applied to an entire row.
- **Quantisation** is a real, non-negligible noise source at the bottom of the
  range.

- **A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising**
  — Wei, Fu, Yang, Huang (CVPR 2020 oral; TPAMI 2021). The "ELD" model:
  Poisson shot noise + Tukey-lambda read noise + row noise + quantisation, with
  a practical per-camera calibration procedure from flat-field and bias frames.
- **Rethinking Noise Synthesis and Modeling in Raw Denoising** — Zhang, Zhu,
  Zheng, et al. (ICCV 2021). Argues for sampling noise parameters over a
  continuum rather than a handful of calibrated ISOs.
- **Practical Poissonian-Gaussian noise modeling and fitting for single-image
  raw-data** — Foi, Trimeche, Katkovnik, Egiazarian (IEEE TIP 2008). The
  scatter-plot method for estimating `a` and `b` from one image, which is what
  runs when you have no calibration.
- **Learning Physics-Informed Noise Models from Dark Frames** (LED, ICCV 2023)
  and related work extend calibration to learned models — not implemented here,
  but the natural next step.

**Consequence:** `denoiseraw/noise/` implements all four ELD components
(`synth.py`), calibration from bias and flat frames (`calibrate.py`), and the
Foi-style blind estimator (`estimate.py`). Row noise is applied so that packed
planes sharing a *physical* sensor row share the offset — a detail that is easy
to get wrong and that decides whether a model learns to remove real banding.

---

## 3. Variance stabilisation

Classical denoisers assume constant-variance noise, which no sensor provides.
The generalised Anscombe transform fixes that, but inverting it naively
introduces a bias, because the denoiser estimates `E[f(x)]` and `sqrt` is
concave.

- **A closed-form approximation of the exact unbiased inverse of the Anscombe
  variance-stabilizing transformation** — Mäkitalo & Foi (IEEE TIP 2011).
- **Optimal inversion of the generalized Anscombe transformation for
  Poisson-Gaussian noise** — Mäkitalo & Foi (IEEE TIP 2013).

**Consequence:** `denoiseraw/vst.py`, used by the classical path and available
as a conditioning mode for the networks. Measured here: the transform holds the
noise standard deviation at 1.00 across three decades of signal, and the
closed-form inverse reduces bias by roughly 15× versus the algebraic one.

---

## 4. Architectures

The benchmark that matters is SIDD (real photographs, real noise). Published
numbers on SIDD sRGB, for orientation:

| Model | Params | SIDD PSNR |
|---|---:|---:|
| BM3D (no training) | – | 25.65 dB |
| DnCNN | 0.56 M | 23.66 dB |
| MIRNet | 31.8 M | 39.72 dB |
| Uformer-B | 50.9 M | 39.89 dB |
| Restormer | 26.1 M | 40.02 dB |
| NAFNet-width64 | 116 M | 40.30 dB |

- **Simple Baselines for Image Restoration** — Chen, Chu, Zhang, Sun
  (ECCV 2022). NAFNet. Argues that attention, GELU and complex gating are all
  unnecessary: a plain U-Net with a multiplicative gate (SimpleGate) and linear
  channel attention matches or beats transformers far more cheaply. **Our
  default**, because on 20–50 MP files the cost difference decides whether the
  thing is usable.
- **Restormer: Efficient Transformer for High-Resolution Image Restoration** —
  Zamir, Arora, Khan, Hayat, Khan, Yang (CVPR 2022). Applies attention across
  *channels* rather than space, making it linear in pixel count. Included
  because its receptive field behaves differently on the low-frequency chroma
  blotches that high-ISO files show.
- **Practical Deep Raw Image Denoising on Mobile Devices** — Wang et al.
  (ECCV 2020). PMRID: separable-convolution U-Net plus the k-Sigma transform.
  The shape of our `lite` preset.
- Newer directions not implemented: MambaIR / MambaIRv2 and EAMamba
  (state-space models, ICCV 2025), and diffusion-based restoration. Both are
  promising; neither has a decisive, reproducible margin on RAW at the time of
  writing, and both cost considerably more at inference.

**Consequence:** `denoiseraw/models/`. Parameter counts are asserted against
the published configurations in `tests/test_models.py`, so the implementations
cannot silently drift from the papers.

---

## 5. Noise conditioning

A blind denoiser must infer the noise level from texture, which is why blind
models over-smooth clean images and under-clean grainy ones.

- **FFDNet: Toward a Fast and Flexible Solution for CNN-based Image Denoising**
  — Zhang, Zuo, Zhang (IEEE TIP 2018). Feed the noise level in as an extra
  input plane.
- **Toward Convolutional Blind Denoising of Real Photographs** — Guo, Yan,
  Zhang, Zuo, Zhang (CVPR 2019). CBDNet: estimate the noise level, then
  condition on the estimate.

RAW lets us go further than a scalar: the calibrated law gives a *per-pixel*
standard deviation, so the network is told exactly how uncertain each pixel is.

**Consequence:** `denoiseraw/models/wrapper.py`, conditioning mode
`sigma_map` (default). It is also how `--strength` works — asking for more or
less smoothing is just overstating or understating sigma.

---

## 6. Adapting to *your* camera without clean data

A model trained on someone else's sensor leaves artefacts on yours. Collecting
clean/noisy pairs per body is impractical.

- **Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images** —
  Huang, Li, Jia, Lu, Liu (CVPR 2021). Two neighbouring pixels see almost the
  same scene with independent noise, so one can be the training target for the
  other; a regularisation term corrects for the residual scene mismatch.
- Predecessors: **Noise2Noise** (Lehtinen et al., ICML 2018) needs two shots of
  the same scene; **Noise2Void** (Krull et al., CVPR 2019) and **Neighbor2Neighbor**
  need only one.

**Consequence:** `denoiseraw/engine/selfsup.py` and `denoiseraw finetune`. On
packed Bayer the 2×2 sub-sampling cell contains four *same-colour* sites two
sensor pixels apart, which is exactly the neighbour relation the method needs.

---

## 7. Classical baseline

- **Image denoising by sparse 3-D transform-domain collaborative filtering** —
  Dabov, Foi, Katkovnik, Egiazarian (IEEE TIP 2007). BM3D. Still the strongest
  denoiser requiring no training, which makes it the right default before any
  weights exist.

**Consequence:** `denoiseraw/classical/bm3d_lite.py` implements both BM3D
passes with no compiled dependency. Measured against the reference `bm3d`
package on cameraman, ours is 0.7–1.1 dB behind (the reference uses a
bi-orthogonal wavelet in the first pass and is compiled), so
`resolve_backend` prefers that package when it is installed. The numbers are in
the module docstring.

---

## 8. Training details worth the bother

- **Charbonnier loss** — Lai, Huang, Ahuja, Yang (CVPR 2017). A differentiable
  L1; far less highlight-dominated than L2 on data spanning four decades.
- **PSNR loss** — used by NAFNet on SIDD: optimise the metric directly.
- **Geometric self-ensemble** — Timofte, Rothe, Van Gool (CVPR 2016). Average
  the output over the eight flips and rotations for 0.1–0.3 dB, free at
  training time. On packed Bayer this requires permuting the colour planes as
  well as flipping pixels; see `denoiseraw/engine/geometry.py`.
- **EMA of weights** — consistently 0.1–0.2 dB, costs nothing. The decay is
  ramped in (as TensorFlow's `ExponentialMovingAverage` and timm both do):
  with a fixed 0.999 and a residual model initialised to the identity, a short
  run's average never leaves its starting point and the saved checkpoint
  denoises nothing at all.
