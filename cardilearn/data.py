"""Dataset contracts and loading helpers.

CardiLearn treats the target and grouping variables as first-class metadata. This makes
it possible to keep patient/donor/animal/study boundaries intact during model development.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    """Validated rectangular dataset used by a training run."""

    frame: pd.DataFrame
    target_column: str
    group_column: str | None = None

    def __post_init__(self) -> None:
        if self.frame.empty:
            raise ValueError("dataset is empty")
        if self.target_column not in self.frame.columns:
            raise KeyError(f"target column not found: {self.target_column}")
        if self.group_column is not None and self.group_column not in self.frame.columns:
            raise KeyError(f"group column not found: {self.group_column}")

    @property
    def target(self) -> pd.Series:
        return self.frame[self.target_column]

    @property
    def groups(self) -> pd.Series | None:
        return None if self.group_column is None else self.frame[self.group_column]

    @property
    def feature_columns(self) -> list[str]:
        excluded = {self.target_column}
        if self.group_column:
            excluded.add(self.group_column)
        return [column for column in self.frame.columns if column not in excluded]

    def features(self) -> pd.DataFrame:
        return self.frame[self.feature_columns].copy()


def load_csv(
    path: str | Path,
    target_column: str,
    group_column: str | None = None,
) -> Dataset:
    """Load a CSV and immediately validate its CardiLearn dataset contract."""

    frame = pd.read_csv(path)
    return Dataset(frame=frame, target_column=target_column, group_column=group_column)
