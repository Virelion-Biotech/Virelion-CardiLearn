"""Conservative expression augmentations for self-supervised prototype training."""
from __future__ import annotations

import torch


def mask_expression(x: torch.Tensor, fraction: float = 0.15) -> torch.Tensor:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    mask = torch.rand_like(x) < fraction
    out = x.clone()
    out[mask] = 0.0
    return out


def make_two_views(x: torch.Tensor, fraction: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    return mask_expression(x, fraction), mask_expression(x, fraction)
