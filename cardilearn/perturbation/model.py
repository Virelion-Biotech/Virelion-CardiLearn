"""Neural predictor for perturbation-induced latent-state changes."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PerturbationPrediction:
    delta_mean: torch.Tensor
    delta_logvar: torch.Tensor

    @property
    def predicted_state(self) -> torch.Tensor:
        raise AttributeError(
            "A perturbation prediction contains a delta; use the baseline state "
            "explicitly with apply_predicted_delta()."
        )


class PerturbationPredictor(nn.Module):
    """Predict a latent response from baseline state and intervention metadata.

    The model predicts a heteroscedastic Gaussian distribution over the latent
    change ``z_perturbed - z_baseline``. It deliberately does not infer a
    perturbation from post-treatment data and does not claim causal effects.
    """

    def __init__(
        self,
        latent_dim: int,
        n_perturbations: int,
        n_perturbation_types: int,
        metadata_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or n_perturbations < 1 or n_perturbation_types < 1:
            raise ValueError("latent_dim and category counts must be positive")
        self.latent_dim = latent_dim
        self.perturbation_embedding = nn.Embedding(n_perturbations, metadata_dim // 2)
        self.type_embedding = nn.Embedding(n_perturbation_types, metadata_dim // 4)
        numeric_dim = 2
        combined_dim = latent_dim + metadata_dim // 2 + metadata_dim // 4 + numeric_dim
        self.context = nn.Sequential(
            nn.Linear(combined_dim, metadata_dim),
            nn.LayerNorm(metadata_dim),
            nn.GELU(),
            nn.Linear(metadata_dim, metadata_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + metadata_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_mean = nn.Linear(hidden_dim, latent_dim)
        self.delta_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        baseline_z: torch.Tensor,
        perturbation_id: torch.Tensor,
        perturbation_type: torch.Tensor,
        dose: torch.Tensor,
        duration: torch.Tensor,
    ) -> PerturbationPrediction:
        if baseline_z.ndim != 2 or baseline_z.shape[1] != self.latent_dim:
            raise ValueError(f"baseline_z must have shape [batch, {self.latent_dim}]")
        batch = baseline_z.shape[0]
        for name, value in (
            ("perturbation_id", perturbation_id),
            ("perturbation_type", perturbation_type),
            ("dose", dose),
            ("duration", duration),
        ):
            if value.ndim != 1 or value.shape[0] != batch:
                raise ValueError(f"{name} must have shape [batch]")

        pid = self.perturbation_embedding(perturbation_id.long())
        ptype = self.type_embedding(perturbation_type.long())
        numeric = torch.stack((dose.float(), duration.float()), dim=-1)
        context = self.context(torch.cat((baseline_z, pid, ptype, numeric), dim=-1))
        hidden = self.trunk(torch.cat((baseline_z, context), dim=-1))
        return PerturbationPrediction(
            delta_mean=self.delta_mean(hidden),
            delta_logvar=self.delta_logvar(hidden).clamp(min=-12.0, max=8.0),
        )
