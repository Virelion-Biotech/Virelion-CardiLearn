"""Data loaders for common public cardiac dataset formats."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import pandas as pd

SUPPORTED = {".csv", ".tsv", ".parquet", ".feather"}

def load_table(path: str | Path, *, sep: str | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(p)
    if p.suffix.lower() not in SUPPORTED: raise ValueError(f"unsupported table format: {p.suffix}")
    if p.suffix.lower() == ".csv": return pd.read_csv(p)
    if p.suffix.lower() == ".tsv": return pd.read_csv(p, sep=sep or "\t")
    if p.suffix.lower() == ".parquet": return pd.read_parquet(p)
    return pd.read_feather(p)

def load_feature_matrix(path: str | Path, target: str, *, id_columns: Iterable[str] = ()) -> tuple[pd.DataFrame, pd.Series]:
    df = load_table(path)
    missing = [c for c in [target, *id_columns] if c not in df.columns]
    if missing: raise ValueError(f"missing required columns: {missing}")
    y = df[target].copy()
    excluded = {target, *id_columns}
    return df.drop(columns=sorted(excluded)), y

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace(" ", "_") for c in out.columns]
    return out.loc[:, ~out.columns.duplicated()].copy()
