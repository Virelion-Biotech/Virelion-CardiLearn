"""Reusable objectives and explicit multi-objective schedules for CardiLearn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn import functional as F


def negative_binomial_nll(
    mu: torch.Tensor,
    theta: torch.Tensor,
    counts: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Stable negative-binomial negative log-likelihood for count data."""
    if mu.shape != theta.shape or mu.shape != counts.shape:
        raise ValueError("mu, theta and counts must have identical shapes")
    if torch.any(counts < 0) or torch.any(~torch.isfinite(counts)):
        raise ValueError("counts must be finite and non-negative")
    if torch.any(mu <= 0) or torch.any(theta <= 0):
        raise ValueError("mu and theta must be positive")
    mu32 = mu.float()
    theta32 = theta.float()
    counts32 = counts.float()
    log_theta_mu = torch.log(theta32 + mu32)
    log_prob = (
        torch.lgamma(counts32 + theta32)
        - torch.lgamma(theta32)
        - torch.lgamma(counts32 + 1.0)
        + theta32 * (torch.log(theta32) - log_theta_mu)
        + counts32 * (torch.log(mu32) - log_theta_mu)
    )
    nll = -log_prob
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction != "mean":
        raise ValueError("reduction must be mean, sum or none")
    return nll.mean()


def masked_log1p_loss(
    prediction: torch.Tensor,
    counts: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != counts.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, counts and mask must have identical shapes")
    target = torch.log1p(torch.clamp(counts, min=0))
    pointwise = F.smooth_l1_loss(prediction, target, reduction="none")
    weights = mask.to(pointwise.dtype)
    return (pointwise * weights).sum() / weights.sum().clamp_min(1.0)


def cosine_alignment_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE for true matched observations only."""
    if z_a.ndim != 2 or z_b.ndim != 2 or z_a.shape != z_b.shape:
        raise ValueError("paired representations must have identical [batch, dim] shapes")
    if z_a.shape[0] < 2:
        raise ValueError("contrastive alignment requires at least two matched observations")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    a = F.normalize(z_a, dim=-1)
    b = F.normalize(z_b, dim=-1)
    logits = (a @ b.T) / temperature
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """VICReg-style loss that separates invariance from collapse prevention."""
    if z_a.shape != z_b.shape or z_a.ndim != 2:
        raise ValueError("z_a and z_b must have identical [batch, dim] shapes")
    if z_a.shape[0] < 2:
        raise ValueError("VICReg requires at least two observations")
    invariance = F.mse_loss(z_a, z_b)
    def variance_term(z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-04)
        return F.relu(1.0 - std).mean()
    def covariance_term(z: torch.Tensor) -> torch.Tensor:
        centered = z - z.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(z.shape[0] - 1, 1)
        off = cov - torch.diag_embed(torch.diagonal(cov))
        return off.pow(2).mean()
    variance = variance_term(z_a) + variance_term(z_b)
    covariance = covariance_term(z_a) + covariance_term(z_b)
    total = (
        invariance_weight * invariance
        + variance_weight * variance
        + covariance_weight * covariance
    )
    return total, {"invariance": invariance, "variance": variance, "covariance": covariance}


@dataclass(frozen=True)
class ObjectiveWeights:
    reconstruction: float = 1.0
    masked: float = 1.0
    contrastive: float = 0.25
    vicreg: float = 0.10
    cell_type: float = 0.50
    maturation: float = 1.0
    injury: float = 0.50
    species_adversarial: float = 0.0

    def __post_init__(self) -> None:
        values = self.__dict__.values()
        if any(value < 0 for value in values):
            raise ValueError("objective weights must be non-negative")


@dataclass(frozen=True)
class ObjectiveStage:
    name: str
    weights: ObjectiveWeights


class ObjectiveSchedule:
    """Explicit curriculum; stages are declarative and reproducible."""

    def __init__(self, stages: tuple[ObjectiveStage, ...]) -> None:
        if not stages:
            raise ValueError("at least one objective stage is required")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("objective stage names must be unique")
        self.stages = stages

    def at_epoch(self, epoch: int) -> ObjectiveStage:
        if epoch < 1:
            raise ValueError("epoch must be >= 1")
        index = min(epoch - 1, len(self.stages) - 1)
        return self.stages[index]

    def to_dict(self) -> dict[str, object]:
        return {
            "stages": [
                {"name": stage.name, "weights": stage.weights.__dict__}
                for stage in self.stages
            ]
        }
