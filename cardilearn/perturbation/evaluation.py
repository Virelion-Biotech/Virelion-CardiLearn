"""Leakage-safe evaluation helpers for perturbation prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_perturbation_predictions(
    predicted_delta: np.ndarray,
    observed_delta: np.ndarray,
) -> pd.DataFrame:
    """Return aggregate latent-response metrics without fitting any model."""
    predicted = np.asarray(predicted_delta, dtype=float)
    observed = np.asarray(observed_delta, dtype=float)
    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError("predicted_delta and observed_delta must have equal 2-D shapes")
    error = predicted - observed
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    pred_norm = np.linalg.norm(predicted, axis=1)
    obs_norm = np.linalg.norm(observed, axis=1)
    denominator = pred_norm * obs_norm
    valid = denominator > 1e-12
    cosine = float(np.mean(np.sum(predicted[valid] * observed[valid], axis=1) / denominator[valid])) if np.any(valid) else 0.0
    direction_accuracy = float(
        np.mean(np.sum(predicted[valid] * observed[valid], axis=1) > 0)
    ) if np.any(valid) else 0.0
    return pd.DataFrame(
        {
            "metric": ["delta_mse", "delta_mae", "cosine_similarity", "direction_accuracy"],
            "value": [mse, mae, cosine, direction_accuracy],
        }
    )
