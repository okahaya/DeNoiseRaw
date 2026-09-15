"""D4 transforms on packed Bayer must match the equivalent mosaic transform."""

import numpy as np
import pytest
import torch

from denoiseraw.engine.geometry import apply_transform, invert_transform, transform_sequences
from denoiseraw.rawio.packing import pack_bayer

SEQS = transform_sequences()


def test_group_has_eight_elements():
    assert len(SEQS) == 8
    assert len(set(SEQS)) == 8


@pytest.mark.parametrize("seq", SEQS)
def test_roundtrip_numpy(seq):
    x = np.random.rand(4, 16, 20).astype(np.float32)
    assert np.allclose(invert_transform(apply_transform(x, seq), seq), x)


@pytest.mark.parametrize("seq", SEQS)
def test_roundtrip_torch(seq):
    x = torch.rand(2, 4, 16, 20)
    assert torch.allclose(invert_transform(apply_transform(x, seq), seq), x)


@pytest.mark.parametrize("generator,mosaic_op", [
    ("hflip", lambda m: np.flip(m, -1)),
    ("vflip", lambda m: np.flip(m, -2)),
    ("transpose", lambda m: np.swapaxes(m, -1, -2)),
])
def test_packed_transform_matches_mosaic_transform(generator, mosaic_op):
    """The whole point: transforming packed planes must equal packing the
    transformed mosaic, channel permutation included."""
    mosaic = np.random.rand(32, 40).astype(np.float32)
    packed = pack_bayer(mosaic, "RGGB")
    assert np.allclose(apply_transform(packed, (generator,)),
                       pack_bayer(np.ascontiguousarray(mosaic_op(mosaic)), "RGGB"))


def test_channel_permutation_is_actually_required():
    """A spatial flip alone is wrong -- guards against 'simplifying' the code."""
    mosaic = np.random.rand(16, 16).astype(np.float32)
    packed = pack_bayer(mosaic, "RGGB")
    naive = np.flip(packed, -1)
    correct = pack_bayer(np.ascontiguousarray(np.flip(mosaic, -1)), "RGGB")
    assert not np.allclose(naive, correct)


def test_non_packed_mode_skips_permutation():
    x = np.random.rand(3, 8, 8).astype(np.float32)
    assert np.allclose(apply_transform(x, ("hflip",), packed=False), np.flip(x, -1))
