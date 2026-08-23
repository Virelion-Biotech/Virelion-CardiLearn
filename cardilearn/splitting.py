"""Leakage-safe deterministic splitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    train_test_split,
)

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


def make_classification_splitter(
    n_splits: int,
    groups: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    random_state: int = 42,
) -> StratifiedGroupKFold:
    """Create a feasibility-checked StratifiedGroupKFold for classification.

    Biological samples are the independent units for cardiac cell-level benchmarks.
    StratifiedGroupKFold keeps every sample intact while attempting to preserve class
    proportions. The explicit feasibility check prevents invalid folds where AUROC
    would be undefined because a validation fold contains only one class.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    group_values = pd.Series(groups).reset_index(drop=True)
    target = pd.Series(y).reset_index(drop=True)
    if len(group_values) != len(target):
        raise ValueError("groups and y must have the same number of observations")
    if group_values.isna().any():
        raise ValueError("group labels cannot contain missing values")
    if target.isna().any():
        raise ValueError("classification targets cannot contain missing values")
    if target.nunique() != 2:
        raise ValueError("MI-vs-Sham classification requires exactly two classes")
    groups_per_class = target.groupby(target).apply(
        lambda values: group_values[values.index].nunique()
    )
    insufficient = groups_per_class[groups_per_class < n_splits]
    if not insufficient.empty:
        detail = ", ".join(f"{cls}={int(count)}" for cls, count in groups_per_class.items())
        raise ValueError(
            f"cannot create {n_splits} stratified grouped folds: each class needs at least "
            f"{n_splits} biological groups ({detail})"
        )
    if group_values.nunique() < n_splits:
        raise ValueError(
            f"cannot create {n_splits} grouped folds from only "
            f"{group_values.nunique()} biological groups"
        )
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


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
