"""The variance-stabilising transform must actually stabilise, and invert
without bias."""

import numpy as np
import pytest
import torch

from denoiseraw.vst import gat, gat_inverse, gat_inverse_algebraic

A, B = 3e-4, 1.6e-5


def test_algebraic_inverse_is_exact():
    x = np.linspace(0, 1, 17)
    assert np.allclose(gat_inverse_algebraic(gat(x, A, B), A, B), x, atol=1e-12)


@pytest.mark.parametrize("level", [0.0, 0.005, 0.05, 0.3, 0.9])
def test_variance_is_stabilised_to_one(level):
    rng = np.random.default_rng(0)
    clean = np.full(200_000, level)
    noisy = rng.poisson(clean / A) * A + rng.normal(0, np.sqrt(B), clean.shape)
    assert gat(noisy, A, B).std() == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("level", [0.002, 0.01, 0.05, 0.3])
def test_unbiased_inverse_beats_the_algebraic_one(level):
    """After denoising we hold E[f(x)], not f(E[x]); sqrt is concave, so the
    algebraic inverse is biased low. The closed form corrects it."""
    rng = np.random.default_rng(1)
    clean = np.full(400_000, level)
    noisy = rng.poisson(clean / A) * A + rng.normal(0, np.sqrt(B), clean.shape)
    z = gat(noisy, A, B).mean()          # a perfect denoiser in the GAT domain
    algebraic_bias = abs(gat_inverse_algebraic(z, A, B) - level)
    unbiased_bias = abs(gat_inverse(z, A, B) - level)
    assert unbiased_bias < algebraic_bias


def test_torch_and_numpy_agree():
    x = np.linspace(0.001, 1.0, 11)
    t = torch.tensor(x, dtype=torch.float64)
    assert np.allclose(gat(t, A, B).numpy(), gat(x, A, B))
    assert np.allclose(gat_inverse(gat(t, A, B), A, B).numpy(), gat_inverse(gat(x, A, B), A, B))
