"""Architectures: shapes, parameter counts, identity init, checkpoints."""

import pytest
import torch

from denoiseraw.models import (
    PRESETS,
    build_model,
    load_checkpoint,
    parameter_count,
    save_checkpoint,
)

CONDITIONINGS = ["sigma_map", "vst", "none"]


@pytest.fixture
def noise_params():
    return torch.full((1, 4, 1, 1), 3e-4), torch.full((1, 4, 1, 1), 1.6e-5)


@pytest.mark.parametrize("preset", sorted(PRESETS))
@pytest.mark.parametrize("conditioning", CONDITIONINGS)
def test_forward_preserves_shape(preset, conditioning, noise_params):
    if preset in ("nafnet", "nafnet-large", "restormer"):
        pytest.skip("large preset; covered by the small variants")
    a, b = noise_params
    model = build_model(preset, conditioning=conditioning).eval()
    x = torch.rand(1, 4, 64, 64) * 0.3
    with torch.no_grad():
        assert model(x, a, b).shape == x.shape


@pytest.mark.parametrize("preset", ["lite", "nafnet-small", "restormer-small"])
def test_residual_models_start_as_the_identity(preset, noise_params):
    """Zero-initialised output layer: the first prediction is the input, so
    training starts from 'do nothing' instead of unlearning random noise."""
    a, b = noise_params
    model = build_model(preset, residual=True).eval()
    x = torch.rand(1, 4, 32, 32) * 0.4
    with torch.no_grad():
        assert torch.allclose(model(x, a, b), x, atol=1e-6)


@pytest.mark.parametrize("size", [(53, 71), (33, 33), (128, 96)])
def test_odd_sizes_are_padded_and_cropped_back(size, noise_params):
    a, b = noise_params
    model = build_model("lite").eval()
    x = torch.rand(1, 4, *size) * 0.2
    with torch.no_grad():
        assert model(x, a, b).shape == x.shape


def test_published_parameter_counts():
    """Guards the architectures against silent drift from the papers."""
    assert parameter_count(build_model("nafnet", conditioning="none", channels=3)) == \
        pytest.approx(29.16e6, rel=0.01)       # NAFNet-width32
    assert parameter_count(build_model("restormer", conditioning="none", channels=3)) == \
        pytest.approx(26.13e6, rel=0.01)       # Restormer denoising config


def test_checkpoint_roundtrip(tmp_path, noise_params):
    a, b = noise_params
    model = build_model("nafnet-small", conditioning="vst").eval()
    x = torch.rand(1, 4, 32, 32) * 0.3
    path = str(tmp_path / "m.ckpt")
    save_checkpoint(path, model, note="hello")

    restored, payload = load_checkpoint(path)
    assert payload["note"] == "hello"
    assert payload["config"]["preset"] == "nafnet-small"
    with torch.no_grad():
        assert torch.allclose(model(x, a, b), restored(x, a, b))


def test_sigma_map_conditioning_requires_parameters():
    model = build_model("lite", conditioning="sigma_map")
    with pytest.raises(ValueError, match="requires the noise parameters"):
        model(torch.rand(1, 4, 32, 32))


def test_unknown_preset_and_conditioning_rejected():
    with pytest.raises(KeyError):
        build_model("does-not-exist")
    with pytest.raises(ValueError, match="unknown conditioning"):
        build_model("lite", conditioning="telepathy")


def test_model_output_responds_to_the_noise_level(noise_params):
    """A conditioned model must actually use its sigma input."""
    model = build_model("lite", residual=False).eval()
    x = torch.rand(1, 4, 32, 32) * 0.3
    a, b = noise_params
    with torch.no_grad():
        low = model(x, a * 0.01, b * 0.01)
        high = model(x, a * 100, b * 100)
    assert not torch.allclose(low, high)
