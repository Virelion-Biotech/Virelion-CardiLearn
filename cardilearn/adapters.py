"""Lightweight adapters for common real cardiac data representations.

The adapters normalize public-dataset exports into a common sample x feature table while
keeping provenance and modality metadata explicit. They deliberately do not download data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModalityTable:
    frame: pd.DataFrame
    modality: str
    source: str
    sample_column: str
    feature_columns: tuple[str, ...]

    @property
    def sample_ids(self) -> pd.Series:
        return self.frame[self.sample_column]


def load_feature_table(path: str | Path, *, modality: str, sample_column: str = "sample_id") -> ModalityTable:
    frame = pd.read_csv(path)
    return make_feature_table(frame, modality=modality, source=str(path), sample_column=sample_column)


def make_feature_table(
    frame: pd.DataFrame,
    *,
    modality: str,
    source: str,
    sample_column: str = "sample_id",
    drop_columns: Iterable[str] = (),
) -> ModalityTable:
    if sample_column not in frame.columns:
        raise KeyError(f"missing sample column: {sample_column}")
    if frame[sample_column].isna().any():
        raise ValueError("sample IDs cannot contain missing values")
    if frame[sample_column].duplicated().any():
        raise ValueError("sample IDs must be unique within a modality table")
    cleaned = frame.drop(columns=[c for c in drop_columns if c in frame], errors="ignore").copy()
    features = tuple(c for c in cleaned.columns if c != sample_column)
    if not features:
        raise ValueError("modality table has no feature columns")
    return ModalityTable(cleaned, modality, source, sample_column, features)


def merge_modalities(
    tables: Iterable[ModalityTable],
    *,
    how: str = "inner",
) -> pd.DataFrame:
    """Align modalities by sample ID; duplicate IDs are rejected before merging."""
    tables = list(tables)
    if not tables:
        raise ValueError("at least one modality table is required")
    result: pd.DataFrame | None = None
    for table in tables:
        frame = table.frame.rename(
            columns={c: f"{table.modality}__{c}" for c in table.feature_columns}
        )
        result = frame if result is None else result.merge(
            frame, on=table.sample_column, how=how, validate="one_to_one"
        )
    assert result is not None
    return result


def expression_log1p(frame: pd.DataFrame, *, exclude: Iterable[str] = ("sample_id",)) -> pd.DataFrame:
    """Safe log1p transform for non-negative molecular abundance tables."""
    excluded = set(exclude)
    result = frame.copy()
    cols = [c for c in result.columns if c not in excluded]
    if (result[cols].select_dtypes(include=[np.number]) < 0).any().any():
        raise ValueError("expression_log1p requires non-negative numeric values")
    result[cols] = np.log1p(result[cols])
    return result
