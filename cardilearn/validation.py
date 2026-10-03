"""Scientific validation checks for dataset and experiment integrity."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class IntegrityReport:
    n_rows: int
    n_features: int
    n_groups: int
    missing_fraction: float
    duplicate_ids: int
    target_missing: int
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool: return not any(w.startswith("ERROR") for w in self.warnings)

def validate_dataset(df: pd.DataFrame, *, target: str, id_column: str = "sample_id", group_column: str | None = None) -> IntegrityReport:
    warnings: list[str] = []
    if target not in df.columns: warnings.append(f"ERROR: target column '{target}' is absent")
    if id_column not in df.columns: warnings.append(f"ERROR: id column '{id_column}' is absent")
    missing = float(df.isna().mean().mean()) if len(df.columns) else 0.0
    duplicate_ids = int(df[id_column].duplicated().sum()) if id_column in df else 0
    target_missing = int(df[target].isna().sum()) if target in df else len(df)
    n_groups = int(df[group_column].nunique()) if group_column and group_column in df else len(df)
    if duplicate_ids: warnings.append(f"WARNING: {duplicate_ids} duplicate sample IDs")
    if target_missing: warnings.append(f"ERROR: {target_missing} rows have missing target values")
    if missing > 0.25: warnings.append(f"WARNING: overall missingness is {missing:.1%}")
    if group_column is None: warnings.append("WARNING: no group column supplied; subject-level leakage may be possible")
    elif group_column not in df.columns: warnings.append(f"ERROR: group column '{group_column}' is absent")
    return IntegrityReport(len(df), max(0, len(df.columns) - 1), n_groups, missing, duplicate_ids, target_missing, tuple(warnings))

def assert_no_group_overlap(train: pd.DataFrame, test: pd.DataFrame, group_column: str | None) -> None:
    if group_column is None or group_column not in train or group_column not in test: return
    overlap = set(train[group_column].dropna()) & set(test[group_column].dropna())
    if overlap: raise AssertionError(f"group leakage detected: {len(overlap)} groups overlap")
