"""Training objectives for perturbation-response prediction."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def delta_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error on latent response vectors."""
    return F.mse_loss(prediction, target)


def direction_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine-direction loss; useful when response magnitude is noisy."""
    pred = F.normalize(prediction, dim=-1, eps=eps)
    truth = F.normalize(target, dim=-1, eps=eps)
    return (1.0 - (pred * truth).sum(dim=-1)).mean()


def gaussian_delta_nll(
    delta_mean: torch.Tensor,
    delta_logvar: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log likelihood for latent deltas."""
    if delta_mean.shape != target.shape or delta_logvar.shape != target.shape:
        raise ValueError("delta_mean, delta_logvar, and target must have equal shapes")
    log_two_pi = torch.log(
        torch.tensor(2.0 * torch.pi, device=target.device, dtype=target.dtype)
    )
    precision_term = (target - delta_mean).pow(2) * torch.exp(-delta_logvar)
    return 0.5 * (delta_logvar + precision_term + log_two_pi).mean()
