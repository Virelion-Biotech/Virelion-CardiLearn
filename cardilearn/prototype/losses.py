"""Training objectives for CardiLearn.

The research path uses likelihoods appropriate for count-based single-cell RNA
measurements rather than treating reconstruction as ordinary Gaussian error.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Legacy smooth-L1 reconstruction used by the prototype."""
    return F.smooth_l1_loss(prediction, target)


def negative_binomial_loss(mu: torch.Tensor, theta: torch.Tensor, counts: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Negative-binomial NLL for UMI/read count data.

    ``mu`` is the expected count and ``theta`` is inverse-dispersion. The
    likelihood explicitly models overdispersion instead of assuming Gaussian
    residuals. This is the default reconstruction objective for CardiLearnLarge.
    """
    if torch.any(counts < 0):
        raise ValueError("counts must be nonnegative")
    if torch.any(mu <= 0) or torch.any(theta <= 0):
        raise ValueError("mu and theta must be positive")
    probs = theta / (theta + mu)
    distribution = torch.distributions.NegativeBinomial(total_count=theta, probs=probs)
    nll = -distribution.log_prob(counts)
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    return nll.mean()


def masked_gene_loss(prediction: torch.Tensor, target_counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Self-supervised masked-gene objective on log1p expression."""
    target = torch.log1p(torch.clamp(target_counts, min=0))
    per_gene = F.smooth_l1_loss(prediction, target, reduction="none")
    mask = mask.to(dtype=per_gene.dtype)
    denom = torch.clamp(mask.sum(), min=1.0)
    return (per_gene * mask).sum() / denom


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    return F.mse_loss(z1, z2)


def variance_loss(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(target_std - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    centered = z - z.mean(dim=0, keepdim=True)
    denom = max(centered.shape[0] - 1, 1)
    covariance = centered.T @ centered / denom
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return (off_diagonal ** 2).mean()


def anti_collapse_loss(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * variance_loss(z) + 0.5 * covariance_loss(z)


def maturation_absolute_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target)


def maturation_pair_loss(score_a: torch.Tensor, score_b: torch.Tensor) -> torch.Tensor:
    return F.softplus(-(score_a - score_b)).mean()


def injury_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target.float())


def cell_type_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target.long())


def species_adversarial_loss(logits: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
    """Classifier loss paired with gradient reversal in CardiLearnLarge."""
    return F.cross_entropy(logits, species.long())


def regeneration_rank_loss(score_a: torch.Tensor, score_b: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    raw = F.softplus(-(score_a - score_b))
    if weight is not None:
        raw = raw * weight
    return raw.mean()
