"""Modality-agnostic, leakage-safe feature extraction primitives."""
from __future__ import annotations
import numpy as np
import pandas as pd

def numeric_summary(X: np.ndarray, prefix: str = "x") -> pd.DataFrame:
    if X.ndim != 2: raise ValueError("X must be 2-dimensional")
    rows = {}
    rows[f"{prefix}_mean"] = np.nanmean(X, axis=1)
    rows[f"{prefix}_std"] = np.nanstd(X, axis=1)
    rows[f"{prefix}_min"] = np.nanmin(X, axis=1)
    rows[f"{prefix}_max"] = np.nanmax(X, axis=1)
    rows[f"{prefix}_median"] = np.nanmedian(X, axis=1)
    return pd.DataFrame(rows)

def waveform_summary(X: np.ndarray) -> pd.DataFrame:
    """Summarize [samples, time] waveforms with stable distributional features."""
    return numeric_summary(X, prefix="waveform")

def expression_variability(X: np.ndarray) -> pd.DataFrame:
    if X.ndim != 2: raise ValueError("expression matrix must be [samples, genes]")
    return pd.DataFrame({"expression_mean": np.mean(X, axis=1), "expression_std": np.std(X, axis=1), "expression_nonzero_fraction": np.mean(X != 0, axis=1)})
