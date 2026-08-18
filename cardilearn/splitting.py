"""Leakage-safe deterministic splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import SplitConfig


@dataclass(frozen=True)
class SplitIndices:
    """Row indices for train/validation/test partitions."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def assert_disjoint(self) -> None:
        sets = [set(self.train.tolist()), set(self.validation.tolist()), set(self.test.tolist())]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise AssertionError("split indices overlap")


def _group_split(
    frame: pd.DataFrame,
    groups: pd.Series,
    config: SplitConfig,
) -> SplitIndices:
    """Perform two group-aware splits so no group crosses a partition."""

    index = np.arange(len(frame))
    first = GroupShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    train_val_pos, test_pos = next(first.split(index, groups=groups.to_numpy()))

    relative_validation = config.validation_size / (1.0 - config.test_size)
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_validation,
        random_state=config.random_state,
    )
    train_pos, val_pos = next(
        second.split(train_val_pos, groups=groups.iloc[train_val_pos].to_numpy())
    )

    result = SplitIndices(
        train=np.sort(train_val_pos[train_pos]),
        validation=np.sort(train_val_pos[val_pos]),
        test=np.sort(test_pos),
    )
    result.assert_disjoint()
    return result


def split_frame(
    frame: pd.DataFrame,
    config: SplitConfig,
    *,
    target: pd.Series | None = None,
    groups: pd.Series | None = None,
) -> SplitIndices:
    """Split rows while preserving groups when provided.

    Stratification is applied only when no group column is supplied. Group-aware splitting
    is deliberately prioritized because avoiding subject-level leakage is more important
    than maintaining exact class proportions.
    """

    if len(frame) == 0:
        raise ValueError("cannot split an empty frame")
    if groups is not None:
        if groups.isna().any():
            raise ValueError("group labels cannot contain missing values")
        if groups.nunique() < 3:
            raise ValueError("group-aware train/validation/test split requires at least 3 groups")
        return _group_split(frame, groups.reset_index(drop=True), config)

    stratify = target if config.stratify and target is not None else None
    all_idx = np.arange(len(frame))
    train_val, test = train_test_split(
        all_idx,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )
    relative_validation = config.validation_size / (1.0 - config.test_size)
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation,
        random_state=config.random_state,
        stratify=None if stratify is None else target.iloc[train_val],
    )
    result = SplitIndices(
        train=np.sort(train), validation=np.sort(validation), test=np.sort(test)
    )
    result.assert_disjoint()
    return result
