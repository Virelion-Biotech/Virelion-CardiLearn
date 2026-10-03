"""Dependency-light ECG/time-series feature extraction."""
from __future__ import annotations

import numpy as np


def summarize_signal(signal, *, sampling_rate_hz: float) -> dict[str, float]:
    """Extract robust morphology-free summary features from a 1-D sampled signal."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("signal must be a 1-D array with at least 3 samples")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    dx = np.diff(x)
    centered = x - np.mean(x)
    return {
        "n_samples": float(x.size),
        "duration_s": float((x.size - 1) / sampling_rate_hz),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "range": float(np.ptp(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
        "mean_abs_derivative": float(np.mean(np.abs(dx))),
        "derivative_std": float(np.std(dx)),
        "zero_crossing_rate": float(np.mean(np.diff(np.signbit(centered)) != 0)),
    }


def batch_summarize(signals, *, sampling_rate_hz: float) -> list[dict[str, float]]:
    return [summarize_signal(signal, sampling_rate_hz=sampling_rate_hz) for signal in signals]
