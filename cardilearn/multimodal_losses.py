"""Losses for training CardiLearn-X without leaking modality identity into Z."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def cross_modal_infonce(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric matched-pair InfoNCE; each row must represent the same unit."""
    if z_a.ndim != 2 or z_b.ndim != 2 or z_a.shape != z_b.shape:
        raise ValueError("paired states must have identical [batch, dim] shapes")
    if z_a.shape[0] < 2:
        raise ValueError("at least two pairs are required for contrastive training")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    a, b = F.normalize(z_a, dim=-1), F.normalize(z_b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def shared_private_orthogonality(z_shared: torch.Tensor, z_private: torch.Tensor) -> torch.Tensor:
    """Penalize linear redundancy between shared and modality-private states."""
    if z_shared.shape[0] != z_private.shape[0]:
        raise ValueError("shared/private states must have the same batch size")
    a = z_shared - z_shared.mean(0, keepdim=True)
    b = z_private - z_private.mean(0, keepdim=True)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return ((a.T @ b) ** 2).mean()


def modality_consistency(states: list[torch.Tensor]) -> torch.Tensor:
    """Mean pairwise distance for matched observations across available modalities."""
    if len(states) < 2:
        return torch.zeros((), device=states[0].device if states else "cpu")
    normalized = [F.normalize(z, dim=-1) for z in states]
    losses = [1.0 - (a * b).sum(-1).mean() for i, a in enumerate(normalized) for b in normalized[i + 1:]]
    return torch.stack(losses).mean()


def uncertainty_regularization(uncertainty: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    """Simple calibration-oriented heteroscedastic objective.

    ``error`` must be an observed held-out/validation residual or absolute
    prediction error; it must never be constructed from the target being used
    to claim prospective performance on the same examples.
    """
    if uncertainty.shape != error.shape:
        raise ValueError("uncertainty and error must have identical shapes")
    log_sigma = torch.log(torch.clamp(uncertainty, min=1e-6))
    return (log_sigma + error / torch.clamp(uncertainty, min=1e-6)).mean()
