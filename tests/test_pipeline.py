"""End-to-end behaviour: the thing must actually reduce noise."""

import os

import numpy as np
import pytest

from denoiseraw.metrics import evaluate_pair, report
from denoiseraw.noise.profile import NoiseProfile
from denoiseraw.pipeline import (
    DenoiseSettings,
    choose_method,
    denoise_file,
    denoise_raw,
    resolve_profile,
    write_outputs,
)
from denoiseraw.rawio.loader import load_raw
from denoiseraw.rawio.packing import pack_bayer


@pytest.mark.parametrize("backend", ["wavelet", "nlm", "bm3d-lite"])
def test_classical_denoising_improves_psnr(synthetic_dng, backend):
    path, clean, profile = synthetic_dng
    settings = DenoiseSettings(method="classical", classical_backend=backend, progress=False)
    result = denoise_raw(load_raw(path), settings, profile=profile)

    clean_packed = pack_bayer(clean, "RGGB")
    before = evaluate_pair(pack_bayer(load_raw(path).data, "RGGB"), clean_packed)
    after = evaluate_pair(pack_bayer(result.raw.data, "RGGB"), clean_packed)
    assert after["psnr"] > before["psnr"] + 1.0
    assert after["ssim"] > before["ssim"]


def test_denoising_does_not_destroy_detail(synthetic_dng):
    """A denoiser that wins PSNR by blurring is not a denoiser."""
    path, clean, profile = synthetic_dng
    result = denoise_raw(load_raw(path),
                         DenoiseSettings(method="classical", classical_backend="bm3d-lite",
                                         progress=False), profile=profile)
    metrics = evaluate_pair(pack_bayer(result.raw.data, "RGGB"), pack_bayer(clean, "RGGB"))
    assert metrics["detail_preservation"] > 0.8


def test_reported_residual_noise_falls(synthetic_dng):
    path, _, profile = synthetic_dng
    result = denoise_raw(load_raw(path),
                         DenoiseSettings(method="classical", classical_backend="wavelet",
                                         progress=False), profile=profile)
    assert result.metrics["residual_noise_after"] < result.metrics["residual_noise_before"]
    assert result.metrics["noise_reduction_db"] > 1.0


def test_method_none_is_a_passthrough(synthetic_dng):
    path, _, profile = synthetic_dng
    img = load_raw(path)
    result = denoise_raw(img, DenoiseSettings(method="none", progress=False), profile=profile)
    assert np.allclose(result.raw.data, img.data)


def test_auto_method_picks_classical_without_weights():
    assert choose_method(DenoiseSettings(method="auto")) == "classical"
    assert choose_method(DenoiseSettings(method="auto", checkpoint="/nope.ckpt")) == "classical"


def test_model_method_without_weights_fails_loudly(synthetic_dng):
    """An untrained network is worse than nothing, so never silently fall back."""
    path, _, profile = synthetic_dng
    with pytest.raises(FileNotFoundError, match="needs trained weights"):
        denoise_raw(load_raw(path),
                    DenoiseSettings(method="model", checkpoint="/does/not/exist.ckpt",
                                    progress=False), profile=profile)


def test_profile_resolution_prefers_the_supplied_file(tmp_path, synthetic_dng):
    path, _, _ = synthetic_dng
    img = load_raw(path)
    profile = NoiseProfile(a=1.234e-4, b=5.6e-6, camera_model="written")
    profile_path = str(tmp_path / "p.json")
    profile.save(profile_path)

    resolved = resolve_profile(img, DenoiseSettings(profile_path=profile_path))
    assert resolved.a == pytest.approx(1.234e-4)

    estimated = resolve_profile(img, DenoiseSettings())
    assert estimated.source == "blind-estimate"


def test_profile_resolution_reads_a_bank_by_iso(tmp_path, synthetic_dng):
    from denoiseraw.noise.profile import ProfileBank

    path, _, _ = synthetic_dng
    bank = ProfileBank("test")
    bank.add(NoiseProfile(a=1e-4, b=4e-6, iso=800))
    bank.add(NoiseProfile(a=4e-4, b=6.4e-5, iso=3200))
    bank_path = str(tmp_path / "bank.json")
    bank.save(bank_path)

    resolved = resolve_profile(load_raw(path),
                               DenoiseSettings(profile_path=bank_path, iso=3200))
    assert resolved.a == pytest.approx(4e-4)


def test_strength_controls_how_hard_it_smooths(synthetic_dng):
    path, _, profile = synthetic_dng
    img = load_raw(path)
    outputs = {}
    for strength in (0.4, 1.6):
        result = denoise_raw(img, DenoiseSettings(method="classical", classical_backend="wavelet",
                                                  strength=strength, progress=False),
                             profile=profile)
        outputs[strength] = result.raw.data
    gentle = report(img.data[None], outputs[0.4][None])["residual_noise_after"]
    strong = report(img.data[None], outputs[1.6][None])["residual_noise_after"]
    assert strong < gentle


@pytest.mark.parametrize("ext", [".dng", ".tif", ".png", ".jpg"])
def test_write_outputs_handles_every_format(tmp_path, synthetic_dng, ext):
    path, _, profile = synthetic_dng
    result = denoise_raw(load_raw(path),
                         DenoiseSettings(method="none", progress=False), profile=profile)
    out = str(tmp_path / f"out{ext}")
    written = write_outputs(result, out, DenoiseSettings())
    assert os.path.exists(next(iter(written.values())))


def test_write_outputs_linear_cfa_mode(tmp_path, synthetic_dng):
    path, _, profile = synthetic_dng
    result = denoise_raw(load_raw(path),
                         DenoiseSettings(method="none", progress=False), profile=profile)
    out = str(tmp_path / "linear.tif")
    written = write_outputs(result, out, DenoiseSettings(extra={"linear_cfa": True}))
    assert "cfa" in written
    assert os.path.exists(str(tmp_path / "linear.json"))


def test_denoise_file_convenience(synthetic_dng):
    path, _, _ = synthetic_dng
    result = denoise_file(path, DenoiseSettings(method="classical",
                                                classical_backend="wavelet", progress=False))
    assert result.method == "classical"
    assert result.elapsed > 0
    assert result.preview().shape[2] == 3


def test_non_bayer_sensors_fall_back_to_single_plane_processing():
    """X-Trans and Quad-Bayer bodies are handled, just without the
    colour-consistency advantage that packing a Bayer mosaic gives."""
    from fixtures import synthetic_scene

    from denoiseraw.rawio.loader import RawImage

    img = RawImage(data=synthetic_scene(128, 128, 0)[..., 1].copy(), pattern=None,
                   black_level=512.0, white_level=16383.0, camera_model="Fujifilm X-T5")
    assert not img.is_bayer
    result = denoise_raw(img, DenoiseSettings(method="classical",
                                              classical_backend="wavelet", progress=False))
    assert result.raw.data.shape == img.data.shape


def test_channel_mismatch_gives_a_useful_error(tmp_path):
    """A 4-plane Bayer model on a 1-plane X-Trans file must say so in English,
    not surface a raw convolution shape error."""
    from fixtures import synthetic_scene

    from denoiseraw.models import build_model, save_checkpoint
    from denoiseraw.rawio.loader import RawImage

    checkpoint = str(tmp_path / "bayer.ckpt")
    save_checkpoint(checkpoint, build_model("lite", channels=4))
    img = RawImage(data=synthetic_scene(64, 64, 0)[..., 1].copy(), pattern=None,
                   black_level=512.0, white_level=16383.0)
    with pytest.raises(ValueError, match="trained for 4-channel data"):
        denoise_raw(img, DenoiseSettings(method="model", checkpoint=checkpoint,
                                         progress=False))


def test_paired_noise_reduction_resists_destroyed_region_artifacts():
    """Regression: independently picking each image's own flattest blocks lets
    a denoiser that destroys texture into near-flatness elsewhere get credited
    as if it had cleaned up the genuinely flat region, which is not what
    happened. Anchoring the flat-block selection to the noisy input alone must
    report the true (lack of) improvement in that region instead."""
    from denoiseraw.metrics import paired_noise_reduction, residual_noise_level

    rng = np.random.default_rng(0)
    h, w = 96, 96
    flat_scene = np.full((h, w), 0.3, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    textured_scene = (0.3 + 0.25 * np.sin(xx / 2.2) * np.cos(yy / 2.7)).astype(np.float32)
    sigma = 0.02
    before = np.stack([np.concatenate([flat_scene, textured_scene], axis=1)
                       + rng.normal(0, sigma, (h, w * 2)).astype(np.float32)])

    after = before.copy()
    # The "denoiser" leaves the truly flat half untouched (no real improvement)
    # but destroys the textured half into near-flatness (texture, not noise).
    after[0, :, w:] = 0.3 + rng.normal(0, 1e-5, (h, w)).astype(np.float32)

    old_before, old_after = residual_noise_level(before), residual_noise_level(after)
    old_db = 20 * np.log10(old_before / max(old_after, 1e-12))
    assert old_db > 20, "the independent-selection method should be fooled here"

    paired = paired_noise_reduction(before, after)
    assert paired["noise_reduction_db"] == pytest.approx(0.0, abs=1.0), (
        "the paired method must report that the genuinely flat region saw no "
        "real improvement, instead of crediting the destroyed texture region"
    )
