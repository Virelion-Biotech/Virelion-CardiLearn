"""Leakage-safe hierarchical splits for CardiLearn prototypes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def study_split(obs: pd.DataFrame, seed: int = 42, train_fraction: float = 0.625, val_fraction: float = 0.125) -> dict[str, list[str]]:
    """Split whole studies; dependent subjects/samples/cells stay together."""
    required = {"study_id", "subject_id", "sample_id"}
    missing = required.difference(obs.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    studies = np.array(sorted(obs["study_id"].unique()), dtype=object)
    if len(studies) < 3:
        raise ValueError("at least three studies are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(studies)
    n_train = max(1, int(round(len(studies) * train_fraction)))
    n_val = max(1, int(round(len(studies) * val_fraction)))
    if n_train + n_val >= len(studies):
        n_val = 1
        n_train = len(studies) - 2
    return {
        "train": sorted(studies[:n_train].tolist()),
        "validation": sorted(studies[n_train:n_train + n_val].tolist()),
        "test": sorted(studies[n_train + n_val:].tolist()),
    }


def assign_split(obs: pd.DataFrame, splits: dict[str, list[str]]) -> pd.DataFrame:
    out = obs.copy()
    lookup = {study: name for name, studies in splits.items() for study in studies}
    out["_split"] = out["study_id"].map(lookup)
    if out["_split"].isna().any():
        raise ValueError("some observations do not belong to a split")
    return out


def assert_no_hierarchy_leakage(obs: pd.DataFrame) -> None:
    """Assert study/subject/sample identities occur in one split only."""
    if "_split" not in obs:
        raise ValueError("expected '_split' column")
    for key in ("study_id", "subject_id", "sample_id"):
        counts = obs.groupby(key)["_split"].nunique()
        if int(counts.max()) > 1:
            leaked = counts[counts > 1].index.tolist()[:10]
            raise AssertionError(f"{key} leakage detected: {leaked}")
