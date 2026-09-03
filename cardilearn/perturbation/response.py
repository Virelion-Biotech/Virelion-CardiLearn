"""Utilities for applying predicted perturbations in latent space."""
from __future__ import annotations

import torch


def apply_predicted_delta(baseline_z: torch.Tensor, delta_z: torch.Tensor) -> torch.Tensor:
    """Construct a predicted perturbed latent state ``z' = z + delta``."""
    if baseline_z.shape != delta_z.shape:
        raise ValueError("baseline_z and delta_z must have identical shapes")
    return baseline_z + delta_z


def predict_counterfactual_state(
    predictor,
    baseline_z: torch.Tensor,
    perturbation_id: torch.Tensor,
    perturbation_type: torch.Tensor,
    dose: torch.Tensor,
    duration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted mean state and one-standard-deviation latent uncertainty."""
    prediction = predictor(
        baseline_z,
        perturbation_id,
        perturbation_type,
        dose,
        duration,
    )
    state = apply_predicted_delta(baseline_z, prediction.delta_mean)
    std = torch.exp(0.5 * prediction.delta_logvar)
    return state, std
