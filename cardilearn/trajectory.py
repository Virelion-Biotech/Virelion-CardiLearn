"""Independent trajectory and temporal generalization checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class TemporalSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    cutoff_time: str
    group_column: str | None


def forward_group_split(
    frame: pd.DataFrame,
    *,
    time_column: str,
    group_column: str | None = None,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> TemporalSplit:
    """Split chronologically while keeping repeated biological groups intact."""
    if time_column not in frame.columns:
        raise ValueError(f"missing time column: {time_column}")
    if not (0 < validation_fraction and 0 < test_fraction and validation_fraction + test_fraction < 1):
        raise ValueError("invalid validation/test fractions")
    times = pd.to_datetime(frame[time_column], errors="coerce")
    if times.isna().any():
        raise ValueError("time column contains unparseable values")

    work = pd.DataFrame({"time": times})
    if group_column is not None:
        if group_column not in frame.columns:
            raise ValueError(f"missing group column: {group_column}")
        work["group"] = frame[group_column].astype(str).to_numpy()
        latest = work.groupby("group")["time"].max().sort_values()
        n_test = max(1, int(np.ceil(len(latest) * test_fraction)))
        n_val = max(1, int(np.ceil(len(latest) * validation_fraction)))
        if n_test + n_val >= len(latest):
            raise ValueError("not enough independent groups for forward train/validation/test split")
        test_groups = set(latest.index[-n_test:])
        val_groups = set(latest.index[-n_test - n_val : -n_test])
        train_mask = ~work["group"].isin(test_groups | val_groups)
        val_mask = work["group"].isin(val_groups)
        test_mask = work["group"].isin(test_groups)
        cutoff = latest.iloc[-n_test - n_val].isoformat()
        return TemporalSplit(
            np.flatnonzero(train_mask.to_numpy()),
            np.flatnonzero(val_mask.to_numpy()),
            np.flatnonzero(test_mask.to_numpy()),
            cutoff,
            group_column,
        )

    order = np.argsort(times.to_numpy(), kind="stable")
    n = len(order)
    n_test = max(1, int(np.ceil(n * test_fraction)))
    n_val = max(1, int(np.ceil(n * validation_fraction)))
    if n_test + n_val >= n:
        raise ValueError("not enough observations for forward train/validation/test split")
    return TemporalSplit(
        np.sort(order[: n - n_val - n_test]),
        np.sort(order[n - n_val - n_test : n - n_test]),
        np.sort(order[n - n_test :]),
        times.iloc[order[n - n_val - n_test]].isoformat(),
        None,
    )


def trajectory_spearman(
    z: np.ndarray,
    stage: Iterable[float],
    *,
    group: Iterable[str] | None = None,
) -> dict[str, float]:
    """Quantify ordered trajectory agreement without fitting a predictive head."""
    values = np.asarray(z, dtype=float)
    stages = np.asarray(list(stage), dtype=float)
    if values.ndim != 2 or len(values) != len(stages):
        raise ValueError("z must be [n, d] and stage must have n entries")
    if not np.isfinite(values).all() or not np.isfinite(stages).all():
        raise ValueError("z and stage must be finite")
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    position = centered @ vh[0]
    if group is None:
        groups = np.asarray(["all"] * len(stages), dtype=object)
    else:
        groups = np.asarray(list(group), dtype=object)
        if len(groups) != len(stages):
            raise ValueError("group must have n entries")

    correlations = []
    for label in pd.unique(groups):
        mask = groups == label
        if mask.sum() < 3 or np.unique(stages[mask]).size < 2:
            continue
        rho = spearmanr(position[mask], stages[mask]).statistic
        if np.isfinite(rho):
            correlations.append(float(rho))
    if not correlations:
        raise ValueError("no group has enough observations with variable stage")
    return {
        "median_spearman": float(np.median(correlations)),
        "median_abs_spearman": float(np.median(np.abs(correlations))),
        "groups_evaluated": float(len(correlations)),
    }
