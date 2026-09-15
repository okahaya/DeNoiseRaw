"""Training, datasets and self-supervised fine-tuning."""

import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from denoiseraw.data import PairedDataset, SyntheticNoiseDataset
from denoiseraw.engine.losses import CharbonnierLoss, CombinedLoss, PSNRLoss
from denoiseraw.engine.selfsup import neighbor2neighbor_loss, neighbor_subsample
from denoiseraw.engine.train import EMA, TrainConfig, evaluate, lr_at, train
from denoiseraw.models import build_model, load_checkpoint
from denoiseraw.noise.profile import NoiseProfile

PROFILE = NoiseProfile(a=5e-4, b=3e-5, sigma_row=1e-3, quantization_step=1 / 16383)


@pytest.fixture(scope="module")
def clean_sources(tmp_path_factory):
    """A handful of pre-packed 'clean RAW' files."""
    d = tmp_path_factory.mktemp("clean")
    rng = np.random.default_rng(0)
    paths = []
    for i in range(6):
        yy, xx = np.mgrid[0:192, 0:192]
        im = 0.15 + 0.3 * (xx / 192) + 0.12 * np.sin(yy / 11.0 + i) + 0.05 * np.cos(xx / 7.0)
        for _ in range(30):
            cy, cx = rng.integers(0, 150), rng.integers(0, 150)
            im[cy:cy + rng.integers(6, 40), cx:cx + rng.integers(6, 40)] += rng.uniform(-0.12, 0.12)
        path = str(d / f"i{i}.npy")
        np.save(path, np.stack([np.clip(im, 0.01, 0.95)] * 4).astype(np.float32))
        paths.append(path)
    return paths, str(d / "cache")


def test_synthetic_dataset_noise_matches_its_own_profile(clean_sources):
    sources, cache = clean_sources
    ds = SyntheticNoiseDataset(sources, patch_size=64, per_image=8,
                               cache_dir=cache + "_a", profile=PROFILE)
    for i in range(len(ds)):
        sample = ds[i]
        clean = sample["clean"].numpy()
        measured = (sample["noisy"].numpy() - clean).std()
        expected = np.sqrt(sample["a"].mean().item() * np.maximum(clean, 0).mean()
                           + sample["b"].mean().item())
        assert measured == pytest.approx(expected, rel=0.15)


def test_dataset_is_deterministic_per_index(clean_sources):
    sources, cache = clean_sources
    ds = SyntheticNoiseDataset(sources, patch_size=64, per_image=4,
                               cache_dir=cache + "_b", profile=PROFILE)
    assert np.allclose(ds[3]["noisy"], ds[3]["noisy"])


def test_paired_dataset_fits_the_true_residual(tmp_path):
    noisy_dir, clean_dir = tmp_path / "n", tmp_path / "c"
    noisy_dir.mkdir()
    clean_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        clean = np.stack([0.05 + 0.4 * (np.mgrid[0:128, 0:128][1] / 128)] * 4).astype(np.float32)
        np.save(str(clean_dir / f"p{i}.npy"), clean)
        np.save(str(noisy_dir / f"p{i}.npy"),
                (clean + rng.normal(0, 0.01, clean.shape)).astype(np.float32))
    sample = PairedDataset(str(noisy_dir), str(clean_dir), patch_size=64)[0]
    assert sample["b"].mean().item() == pytest.approx(1e-4, rel=0.3)


def test_paired_dataset_needs_matching_names(tmp_path):
    (tmp_path / "n").mkdir()
    (tmp_path / "c").mkdir()
    np.save(str(tmp_path / "n" / "a.npy"), np.zeros((4, 8, 8), np.float32))
    np.save(str(tmp_path / "c" / "b.npy"), np.zeros((4, 8, 8), np.float32))
    with pytest.raises(RuntimeError, match="no matching filenames"):
        PairedDataset(str(tmp_path / "n"), str(tmp_path / "c"))


def test_losses_are_zero_for_a_perfect_prediction():
    x = torch.rand(2, 4, 16, 16)
    assert float(CharbonnierLoss(eps=0.0)(x, x)) == pytest.approx(0.0, abs=1e-6)
    assert float(PSNRLoss()(x, x)) < -70          # PSNR loss is negated PSNR
    total, parts = CombinedLoss(charbonnier=1.0, frequency=1.0, shadow=1.0)(x, x)
    assert float(total) == pytest.approx(0.0, abs=3e-3)
    assert set(parts) == {"charbonnier", "frequency", "shadow"}


def test_combined_loss_skips_zero_weighted_terms():
    x, y = torch.rand(1, 4, 8, 8), torch.rand(1, 4, 8, 8)
    _, parts = CombinedLoss(charbonnier=1.0)(x, y)
    assert set(parts) == {"charbonnier"}


def test_lr_schedule_warms_up_then_decays():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-6, warmup_steps=100)
    assert lr_at(0, 1000, cfg) < lr_at(50, 1000, cfg) < lr_at(99, 1000, cfg)
    assert lr_at(100, 1000, cfg) == pytest.approx(1e-3, rel=1e-3)
    assert lr_at(1000, 1000, cfg) == pytest.approx(1e-6, abs=1e-7)


def test_ema_tracks_and_restores():
    model = build_model("lite")
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    backup = ema.copy_to(model)
    ema.restore(model, backup)
    for k, v in model.state_dict().items():
        assert torch.allclose(v, original[k])


def test_neighbor_subsample_never_picks_the_same_pixel_twice():
    x = torch.arange(2 * 4 * 8 * 8, dtype=torch.float32).reshape(2, 4, 8, 8)
    for _ in range(30):
        g1, g2 = neighbor_subsample(x)
        assert not torch.any(g1 == g2)
        assert g1.shape == (2, 4, 4, 4)


def test_neighbor2neighbor_produces_gradients():
    model = build_model("lite")
    a = torch.full((1, 4, 1, 1), 3e-4)
    b = torch.full((1, 4, 1, 1), 1.6e-5)
    loss, parts = neighbor2neighbor_loss(model, torch.rand(2, 4, 64, 64) * 0.3, a, b)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    assert parts["reconstruction"] >= 0 and parts["regularisation"] >= 0


@pytest.mark.slow
def test_training_actually_improves_psnr(clean_sources, tmp_path):
    """The headline check: a short run on CPU must beat the noisy input."""
    torch.manual_seed(0)
    sources, cache = clean_sources
    common = dict(patch_size=64, cache_dir=cache + "_t", profile=PROFILE)
    train_set = SyntheticNoiseDataset(sources, per_image=40, seed=1, **common)
    val_set = SyntheticNoiseDataset(sources, per_image=8, seed=99, augment=False,
                                    length=32, **common)
    model = build_model("lite", overrides={"width": 16, "depth": 3, "blocks_per_level": 1})
    device = torch.device("cpu")
    loader = DataLoader(val_set, batch_size=8)

    at_init = evaluate(model, loader, device)
    assert at_init["psnr"] == pytest.approx(at_init["psnr_input"], abs=0.01), \
        "a residual model must start as the identity"

    cfg = TrainConfig(epochs=25, batch_size=8, lr=1e-3, warmup_steps=50, num_workers=0,
                      out_dir=str(tmp_path / "run"), log_every=100, ema_decay=0.995, val_every=5)
    best = train(model, train_set, val_set, cfg, device=device)
    assert os.path.exists(best)

    trained, _ = load_checkpoint(best)
    after = evaluate(trained, loader, device)
    assert after["gain"] > 1.5, f"training gained only {after['gain']:.2f} dB"


def test_evaluate_handles_an_empty_loader():
    model = build_model("lite")
    empty = DataLoader(SyntheticNoiseDatasetStub(), batch_size=1)
    metrics = evaluate(model, empty, torch.device("cpu"))
    assert np.isnan(metrics["psnr"])


class SyntheticNoiseDatasetStub(torch.utils.data.Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, i):
        raise IndexError


def test_ema_warmup_prevents_a_short_run_saving_an_identity_model():
    """Regression: with a fixed 0.999 decay, a few hundred steps leave the
    average still sitting on its zero-initialised starting point -- so the saved
    checkpoint is an identity function that denoises nothing, which looks like a
    model that merely failed to learn."""
    target = build_model("lite")
    with torch.no_grad():
        for p in target.parameters():
            p.fill_(1.0)

    results = {}
    for warmup in (True, False):
        ema = EMA(build_model("lite"), decay=0.999, warmup=warmup)
        with torch.no_grad():
            for v in ema.shadow.values():
                v.fill_(0.0)
        for _ in range(480):
            ema.update(target)
        results[warmup] = float(next(iter(ema.shadow.values())).mean())

    assert results[True] == pytest.approx(1.0, abs=0.01)
    assert results[False] < 0.5           # what the bug looked like
    assert EMA(target, 0.999).current_decay() < 0.2   # tracks closely at step 0


def test_short_run_warns_that_warmup_swallows_it(clean_sources, tmp_path):
    sources, cache = clean_sources
    train_set = SyntheticNoiseDataset(sources, patch_size=64, per_image=4,
                                      cache_dir=cache + "_w", profile=PROFILE)
    cfg = TrainConfig(epochs=1, batch_size=8, warmup_steps=5000, num_workers=0,
                      out_dir=str(tmp_path / "short"))
    with pytest.warns(RuntimeWarning, match="warmup alone"):
        train(build_model("lite"), train_set, None, cfg, device=torch.device("cpu"))
