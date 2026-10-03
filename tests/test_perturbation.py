from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from cardilearn.perturbation import (
    PerturbationPredictor,
    apply_predicted_delta,
    delta_mse_loss,
    direction_loss,
    evaluate_perturbation_predictions,
    gaussian_delta_nll,
    predict_counterfactual_state,
)


def _inputs(batch: int = 6, latent_dim: int = 8):
    torch.manual_seed(21)
    baseline = torch.randn(batch, latent_dim)
    pid = torch.tensor([0, 1, 2, 0, 1, 2])[:batch]
    ptype = torch.tensor([0, 0, 1, 1, 0, 1])[:batch]
    dose = torch.linspace(0.0, 1.0, batch)
    duration = torch.linspace(1.0, 6.0, batch)
    return baseline, pid, ptype, dose, duration


def test_predictor_shapes_and_uncertainty():
    model = PerturbationPredictor(8, n_perturbations=3, n_perturbation_types=2)
    baseline, pid, ptype, dose, duration = _inputs()
    prediction = model(baseline, pid, ptype, dose, duration)
    assert prediction.delta_mean.shape == (6, 8)
    assert prediction.delta_logvar.shape == (6, 8)
    assert torch.isfinite(prediction.delta_mean).all()


def test_counterfactual_state_is_baseline_plus_delta():
    model = PerturbationPredictor(8, 3, 2)
    baseline, pid, ptype, dose, duration = _inputs()
    prediction = model(baseline, pid, ptype, dose, duration)
    expected = apply_predicted_delta(baseline, prediction.delta_mean)
    state, std = predict_counterfactual_state(model, baseline, pid, ptype, dose, duration)
    assert torch.allclose(state, expected)
    assert torch.all(std > 0)


def test_losses_and_evaluation_are_finite():
    prediction = torch.randn(5, 4)
    target = torch.randn(5, 4)
    logvar = torch.zeros_like(target)
    assert torch.isfinite(delta_mse_loss(prediction, target))
    assert torch.isfinite(direction_loss(prediction, target))
    assert torch.isfinite(gaussian_delta_nll(prediction, logvar, target))
    metrics = evaluate_perturbation_predictions(prediction.numpy(), target.numpy())
    assert list(metrics["metric"]) == ["delta_mse", "delta_mae", "cosine_similarity", "direction_accuracy"]
    assert np.isfinite(metrics["value"]).all()


def test_invalid_shape_is_rejected():
    model = PerturbationPredictor(4, 2, 1)
    baseline, pid, ptype, dose, duration = _inputs(batch=3, latent_dim=4)
    with pytest.raises(ValueError):
        model(baseline[:, :3], pid, ptype, dose, duration)
