"""Temporal evaluation utilities for longitudinal cardiac datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(
    frame: pd.DataFrame,
    *,
    time_column: str,
    test_fraction: float = 0.2,
    group_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a chronological development/test split.

    When a group column is present, all observations from groups whose latest timestamp
    lies beyond the cutoff are kept in the future test partition.
    """
    if time_column not in frame:
        raise KeyError(f"missing time column: {time_column}")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    times = pd.to_datetime(frame[time_column], errors="coerce")
    if times.isna().any():
        raise ValueError("time column contains unparseable timestamps")
    if group_column is None:
        order = np.argsort(times.to_numpy())
        n_test = max(1, int(np.ceil(len(order) * test_fraction)))
        return np.sort(order[:-n_test]), np.sort(order[-n_test:])
    if group_column not in frame:
        raise KeyError(f"missing group column: {group_column}")
    group_last = frame.assign(_time=times).groupby(group_column)["_time"].max().sort_values()
    n_test_groups = max(1, int(np.ceil(len(group_last) * test_fraction)))
    test_groups = set(group_last.index[-n_test_groups:])
    test_mask = frame[group_column].isin(test_groups).to_numpy()
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
