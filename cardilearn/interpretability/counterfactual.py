"""Counterfactual sensitivity tools for CardiLearn explanations."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
import torch


def gene_mask_counterfactual(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    feature_indices: Sequence[int],
    *,
    baseline_value: float = 0.0,
) -> pd.DataFrame:
    """Measure prediction changes after jointly masking selected genes."""
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [samples, genes]")
    indices = list(feature_indices)
    if not indices:
        raise ValueError("feature_indices cannot be empty")
    if any(i < 0 or i >= inputs.shape[1] for i in indices):
        raise IndexError("feature index outside input range")
    with torch.no_grad():
        original = forward_fn(inputs)
        if original.ndim != 1:
            raise ValueError("forward_fn must return one scalar per sample")
        masked = inputs.clone()
        masked[:, indices] = baseline_value
        counterfactual = forward_fn(masked)
    delta = original - counterfactual
    return pd.DataFrame(
        {
            "original_mean": [float(original.mean().item())],
            "counterfactual_mean": [float(counterfactual.mean().item())],
            "delta_mean": [float(delta.mean().item())],
            "n_masked_features": [len(indices)],
        }
    )


def ranked_gene_counterfactuals(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    ranked_feature_indices: Sequence[int],
    *,
    step: int = 1,
    baseline_value: float = 0.0,
) -> pd.DataFrame:
    """Progressively mask a ranked gene list to test explanatory concentration."""
    if step < 1:
        raise ValueError("step must be positive")
    indices = list(ranked_feature_indices)
    if not indices:
        raise ValueError("ranked_feature_indices cannot be empty")
    with torch.no_grad():
        original = forward_fn(inputs)
        rows = []
        for end in range(step, len(indices) + 1, step):
            masked = inputs.clone()
            masked[:, indices[:end]] = baseline_value
            prediction = forward_fn(masked)
            rows.append(
                {
                    "n_masked_features": end,
                    "prediction_mean": float(prediction.mean().item()),
                    "delta_from_original": float((original - prediction).mean().item()),
                }
            )
    return pd.DataFrame(rows)
