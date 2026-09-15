"""The command-line interface is the product; every subcommand must work."""

import json
import os

import numpy as np
import pytest

from denoiseraw.cli import main


def run(*argv) -> int:
    return main([str(a) for a in argv])


def test_info(synthetic_dng, capsys):
    path, _, _ = synthetic_dng
    assert run("info", path) == 0
    out = capsys.readouterr().out
    assert "Canon EOS 5D Mark IV" in out
    assert "CFA=RGGB" in out


def test_estimate_writes_a_profile(synthetic_dng, tmp_path, capsys):
    path, _, _ = synthetic_dng
    profile_path = str(tmp_path / "p.json")
    assert run("estimate", path, "-o", profile_path) == 0
    assert "NoiseProfile" in capsys.readouterr().out
    with open(profile_path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["a"] > 0 and data["b"] > 0


def test_denoise_writes_a_dng(synthetic_dng, tmp_path, capsys):
    path, _, _ = synthetic_dng
    out = str(tmp_path / "out.dng")
    assert run("denoise", path, "-o", out, "--method", "classical",
               "--backend", "wavelet") == 0
    assert os.path.exists(out)
    assert "dB" in capsys.readouterr().out


@pytest.mark.parametrize("ext", [".png", ".tif", ".jpg", ".dng"])
def test_denoise_output_formats(synthetic_dng, tmp_path, ext):
    path, _, _ = synthetic_dng
    out = str(tmp_path / f"out{ext}")
    assert run("denoise", path, "-o", out, "--method", "classical",
               "--backend", "wavelet", "-q") == 0
    assert os.path.getsize(out) > 0


def test_denoise_batch_to_a_directory(synthetic_dng, tmp_path):
    path, _, _ = synthetic_dng
    second = str(tmp_path / "second.dng")
    import shutil
    shutil.copy(path, second)
    out_dir = str(tmp_path / "out")
    assert run("denoise", path, second, "-o", out_dir, "--method", "classical",
               "--backend", "wavelet", "-q") == 0
    assert len([f for f in os.listdir(out_dir) if f.endswith(".dng")]) == 2


def test_denoise_reports_failures_but_keeps_going(synthetic_dng, tmp_path, capsys):
    path, _, _ = synthetic_dng
    broken = str(tmp_path / "broken.dng")
    with open(broken, "wb") as fh:
        fh.write(b"not a raw file")
    rc = run("denoise", path, broken, "-o", str(tmp_path / "o"), "--method", "classical",
             "--backend", "wavelet", "-q")
    assert rc == 1
    assert "FAILED" in capsys.readouterr().err


def test_calibrate_from_bias_frames(tmp_path, capsys):
    from fixtures import CANON_5D4_CAM_XYZ

    from denoiseraw.noise.profile import NoiseProfile
    from denoiseraw.noise.synth import synthesize_noise
    from denoiseraw.rawio.loader import RawImage
    from denoiseraw.rawio.packing import unpack_bayer
    from denoiseraw.rawio.writer import save_dng

    truth = NoiseProfile(a=0.0, b=1.6e-5, sigma_row=2e-3, tukey_lambda=0.14)
    rng = np.random.default_rng(0)
    bias_paths = []
    for i in range(3):
        packed = synthesize_noise(np.zeros((4, 128, 128), np.float32), truth,
                                  model="eld", rng=rng)
        img = RawImage(data=unpack_bayer(packed, "RGGB"), pattern="RGGB",
                       black_level=512.0, white_level=16383.0,
                       color_matrix=CANON_5D4_CAM_XYZ, camera_model="Canon EOS 5D Mark IV")
        p = str(tmp_path / f"bias{i}.dng")
        save_dng(p, img)
        bias_paths.append(p)

    out = str(tmp_path / "profile.json")
    assert run("calibrate", "--bias", *bias_paths, "--iso", 1600, "-o", out) == 0
    assert "without flat frames" in capsys.readouterr().out
    profile = NoiseProfile.load(out)
    assert profile.iso == 1600
    assert profile.b == pytest.approx(truth.b, rel=0.4)


def test_finetune_runs_and_saves(synthetic_dng, tmp_path, capsys):
    path, _, _ = synthetic_dng
    out = str(tmp_path / "ft.ckpt")
    assert run("finetune", path, "--preset", "lite", "--steps", 3,
               "--batch-size", 2, "--patch-size", 64, "-o", out) == 0
    assert os.path.exists(out)
    from denoiseraw.models import load_checkpoint
    _, payload = load_checkpoint(out)
    assert "finetuned_on" in payload


def test_bench_compares_methods(synthetic_dng, capsys):
    path, _, _ = synthetic_dng
    assert run("bench", path, "--crop", 128, "--backends", "wavelet") == 0
    out = capsys.readouterr().out
    assert "(noisy input)" in out and "classical:wavelet" in out


def test_train_reports_a_missing_data_directory(tmp_path, capsys):
    assert run("train", "--data", str(tmp_path / "nothing")) == 1
    assert "no RAW files" in capsys.readouterr().err
