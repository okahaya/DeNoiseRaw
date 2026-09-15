"""Self-supervised fine-tuning on your own noisy RAW files.

The hardest part of deploying a RAW denoiser is that the model was trained on
*somebody else's sensor*. Noise differs between bodies, and a model tuned to a
Sony sensor leaves chroma blotches on a Canon one.

Neighbor2Neighbor (Huang et al., CVPR 2021) fixes this without any clean data.
The idea: two pixels that are spatial neighbours see almost the same scene but
independent noise, so one can serve as a training target for the other. Sample
two disjoint sub-images ``g1(y)``, ``g2(y)`` by picking two different pixels from
each 2x2 cell, and train ``f(g1(y)) -> g2(y)``.

The naive version over-smooths, because the two sub-images are not *exactly* the
same scene. The paper's regularisation term corrects for that mismatch by
measuring it on the network's own output::

    L = ||f(g1) - g2||^2 + gamma * || f(g1) - g2 - (g1(f(y)) - g2(f(y))) ||^2

On packed Bayer the sub-sampling happens inside each plane, so the 2x2 cell is
four *same-colour* sites two sensor pixels apart -- which is exactly the
neighbour relation the method needs.
"""

from __future__ import annotations

from typing import Tuple

import torch


def neighbor_subsample(x: torch.Tensor, generator=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``(B, C, H, W)`` into two disjoint half-resolution sub-images.

    From each 2x2 cell we draw an ordered pair of distinct positions at random,
    sending the first to ``g1`` and the second to ``g2``.
    """
    b, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    x = x[..., : h2 * 2, : w2 * 2]
    # (B, C, h2, w2, 4): the four members of each 2x2 cell.
    cells = (x.reshape(b, c, h2, 2, w2, 2)
              .permute(0, 1, 2, 4, 3, 5)
              .reshape(b, c, h2, w2, 4))

    device = x.device
    # 12 ordered pairs of distinct cell positions.
    pairs = torch.tensor([[i, j] for i in range(4) for j in range(4) if i != j],
                         device=device)
    idx = torch.randint(len(pairs), (b, c, h2, w2), device=device, generator=generator)
    chosen = pairs[idx]                                   # (B, C, h2, w2, 2)

    g1 = torch.gather(cells, -1, chosen[..., 0:1]).squeeze(-1)
    g2 = torch.gather(cells, -1, chosen[..., 1:2]).squeeze(-1)
    return g1, g2


def neighbor2neighbor_loss(
    model,
    noisy: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gamma: float = 2.0,
    generator=None,
) -> Tuple[torch.Tensor, dict]:
    """Neighbor2Neighbor objective for a noise-conditioned model.

    ``a`` / ``b`` are the Poisson-Gaussian parameters of ``noisy``. Because the
    sub-images are drawn from the same sensor at the same exposure, their noise
    statistics are unchanged, so the same parameters apply to the sub-sampled
    inputs.
    """
    g1, g2 = neighbor_subsample(noisy, generator=generator)
    pred = model(g1, a, b)

    diff = pred - g2
    with torch.no_grad():
        # The scene mismatch between g1 and g2, measured on a denoised image
        # where it is not masked by noise.
        denoised_full = model(noisy, a, b)
        d1, d2 = neighbor_subsample(denoised_full, generator=generator)
        correction = d1 - d2

    rec = (diff ** 2).mean()
    reg = ((diff - correction) ** 2).mean()
    total = rec + gamma * reg
    return total, {"reconstruction": float(rec.detach()), "regularisation": float(reg.detach())}
