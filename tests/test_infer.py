"""Tiled inference must not leave seams, and self-ensembling must be coherent."""

import numpy as np
import pytest
import torch

from denoiseraw.engine.infer import denoise_packed, estimate_tile_size, pick_device
from denoiseraw.models import build_model
from denoiseraw.noise.profile import NoiseProfile

PROFILE = NoiseProfile(a=3e-4, b=1.6e-5)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    # residual=False so the (untrained) network does something non-trivial and
    # the tiling machinery is genuinely exercised.
    return build_model("lite", residual=False).eval()


@pytest.fixture(scope="module")
def image():
    return (np.random.default_rng(0).random((4, 200, 260)) * 0.4).astype(np.float32)


@pytest.mark.parametrize("tile,overlap", [(128, 64), (128, 32), (96, 48), (256, 64)])
def test_tiled_matches_untiled(model, image, tile, overlap):
    full = denoise_packed(model, image, PROFILE, tile=0)
    tiled = denoise_packed(model, image, PROFILE, tile=tile, overlap=overlap)
    assert np.abs(tiled - full).mean() < 2e-3


@pytest.mark.parametrize("overlap", [64, 32, 16])
def test_no_step_discontinuity_at_interior_seams(model, image, overlap):
    """The real seam test.

    A crossfade between two disagreeing tiles necessarily raises the gradient a
    little across the *whole* overlap band, and the image's outer border is
    reflect-padded whether or not we tile. Neither is a seam. A seam is a *step*
    at a tile boundary, so compare each interior boundary column against its own
    immediate neighbourhood rather than against the global average.
    """
    tile = 128
    out = denoise_packed(model, image, PROFILE, tile=tile, overlap=overlap)
    untiled = denoise_packed(model, image, PROFILE, tile=0)
    # Excess gradient attributable to tiling, ignoring what the model does anyway.
    excess = (np.abs(np.diff(out.mean(0), axis=1)).mean(0)
              - np.abs(np.diff(untiled.mean(0), axis=1)).mean(0))
    stride = tile - overlap
    margin = 8
    for boundary in range(stride, out.shape[2] - tile, stride):
        local = excess[boundary - margin:boundary + margin + 1]
        neighbourhood = np.concatenate([excess[boundary - 4 * margin:boundary - margin],
                                        excess[boundary + margin + 1:boundary + 4 * margin]])
        assert np.abs(local).max() < np.abs(neighbourhood).max() + 1e-3, (
            f"step discontinuity at tile boundary x={boundary}")


def test_self_ensemble_runs_and_changes_nothing_structurally(model, image):
    out = denoise_packed(model, image, PROFILE, tile=0, self_ensemble=True)
    assert out.shape == image.shape
    assert np.isfinite(out).all()


def test_self_ensemble_is_exact_for_an_equivariant_model(image):
    """A pixel-wise model commutes with every D4 element, so ensembling must
    return exactly the single-pass result. If the channel permutations were
    wrong, the eight estimates would disagree and this would fail."""
    class Identity(torch.nn.Module):
        padder_size = 1

        def forward(self, x, a=None, b=None):
            return x * 0.5

    model = Identity()
    single = denoise_packed(model, image, PROFILE, tile=0)
    ensembled = denoise_packed(model, image, PROFILE, tile=0, self_ensemble=True)
    assert np.allclose(single, ensembled, atol=1e-6)


def test_device_and_tile_helpers():
    assert pick_device("cpu").type == "cpu"
    assert estimate_tile_size(free_bytes=4 << 30) % 64 == 0
    assert estimate_tile_size(free_bytes=1 << 20) >= 128
