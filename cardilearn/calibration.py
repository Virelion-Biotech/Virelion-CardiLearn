"""Calibration and uncertainty helpers for probabilistic classifiers."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import brier_score_loss

def brier_score(y_true, probability) -> float:
    return float(brier_score_loss(y_true, np.asarray(probability)))

def expected_calibration_error(y_true, probability, *, bins=10) -> float:
    y = np.asarray(y_true); p = np.asarray(probability); edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not mask.any(): continue
        ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)
