"""Optional species-adversarial component for CardiLearn.

This module intentionally stays separate from the v0.1 encoder. The adversarial
objective should only be enabled after transfer baselines establish that species
information is harming biological generalization rather than representing
legitimate biology.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coefficient * grad_output, None


class GradientReversal(nn.Module):
    """Identity forward pass with a sign-reversed gradient in backpropagation."""

    def __init__(self, coefficient: float = 1.0) -> None:
        super().__init__()
        if coefficient < 0:
            raise ValueError("coefficient must be non-negative")
        self.coefficient = float(coefficient)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversal.apply(x, self.coefficient)


class SpeciesAdversary(nn.Module):
    """Predict species from a latent representation through gradient reversal."""

    def __init__(
        self,
        latent_dim: int,
        n_species: int,
        hidden_dim: int = 64,
        coefficient: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or n_species < 2:
            raise ValueError("latent_dim must be positive and n_species must be >= 2")
        self.reversal = GradientReversal(coefficient)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_species),
        )

    def forward(self, z_shared: torch.Tensor) -> torch.Tensor:
        if z_shared.ndim != 2:
            raise ValueError(f"expected [batch, latent], got {tuple(z_shared.shape)}")
        return self.classifier(self.reversal(z_shared))
