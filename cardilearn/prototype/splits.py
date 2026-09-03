"""Leakage-safe hierarchical splits for CardiLearn prototypes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def study_split(
    obs: pd.DataFrame,
    seed: int = 42,
    train_fraction: float = 0.625,
    val_fraction: float = 0.125,
    group_column: str = "study_id",
) -> dict[str, list[str]]:
    """Split independent groups while keeping linked studies together.

    ``group_column='study_family_id'`` should be used when a registry has
    identified related accessions that are not independent experiments.
    The returned split names contain group IDs, not individual study IDs.
    """
    required = {"study_id", "subject_id", "sample_id", group_column}
    missing = required.difference(obs.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    groups = np.array(sorted(obs[group_column].dropna().unique()), dtype=object)
    if len(groups) < 3:
        raise ValueError("at least three independent groups are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_train = max(1, round(len(groups) * train_fraction))
    n_val = max(1, round(len(groups) * val_fraction))
    if n_train + n_val >= len(groups):
        n_val = 1
        n_train = len(groups) - 2
    return {
        "train": sorted(groups[:n_train].tolist()),
        "validation": sorted(groups[n_train : n_train + n_val].tolist()),
        "test": sorted(groups[n_train + n_val :].tolist()),
        "group_column": group_column,
    }


def assign_split(obs: pd.DataFrame, splits: dict[str, list[str]], *, group_column: str = "study_id") -> pd.DataFrame:
    """Assign observations to splits using an explicit independence group."""
    if group_column not in obs.columns:
        raise ValueError(f"missing split group column: {group_column}")
    out = obs.copy()
    lookup = {group: name for name, groups in splits.items() if name != "group_column" for group in groups}
    out["_split"] = out[group_column].map(lookup)
    if out["_split"].isna().any():
        raise ValueError("some observations do not belong to a split")
    return out


def assert_no_hierarchy_leakage(obs: pd.DataFrame) -> None:
    """Assert study/subject/sample identities occur in one split only."""
    if "_split" not in obs:
        raise ValueError("expected '_split' column")
    for key in ("study_id", "subject_id", "sample_id"):
        counts = obs.groupby(key)["_split"].nunique()
        if counts.max() > 1:
            leaked = counts[counts > 1].index.tolist()[:10]
            raise AssertionError(f"{key} leakage detected: {leaked}")
    if "study_family_id" in obs.columns:
        counts = obs.groupby("study_family_id")["_split"].nunique()
        if counts.max() > 1:
            leaked = counts[counts > 1].index.tolist()[:10]
            raise AssertionError(f"study_family_id leakage detected: {leaked}")
