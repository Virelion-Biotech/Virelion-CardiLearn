"""Calibration and uncertainty helpers for probabilistic classifiers."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def calibrate(estimator, X, y, *, method: str = "sigmoid", cv: int = 5):
    """Fit a cross-validated calibrated classifier on training data only."""
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be 'sigmoid' or 'isotonic'")
    if cv < 2:
        raise ValueError("cv must be >= 2")
    calibrated = CalibratedClassifierCV(estimator, method=method, cv=cv)
    calibrated.fit(X, y)
    return calibrated


def brier_score(y_true, probability) -> float:
    return float(brier_score_loss(y_true, np.asarray(probability)))


def expected_calibration_error(y_true, probability, *, bins: int = 10) -> float:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    y = np.asarray(y_true)
    p = np.asarray(probability)
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and probability must have the same length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)
