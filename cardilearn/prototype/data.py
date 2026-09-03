"""Prototype dataset utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from cardilearn.leakage import assert_no_leakage


class CardiLearnCellDataset(Dataset):
    """Dense prototype dataset; real large-scale data will use lazy/sparse loading."""

    def __init__(self, X: np.ndarray, metadata: pd.DataFrame):
        required = {"species", "assay", "cell_type", "maturation", "injury"}
        missing = required.difference(metadata.columns)
        if missing:
            raise ValueError(f"missing target/context columns: {sorted(missing)}")
        if X.ndim != 2 or X.shape[0] != len(metadata):
            raise ValueError("X and metadata have incompatible shapes")
        self.X = np.asarray(X, dtype=np.float32)
        self.metadata = metadata.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        return {
            "x": self.X[index],
            "species": int(row["species"]),
            "assay": int(row["assay"]),
            "cell_type": int(row["cell_type"]),
            "maturation": float(row["maturation"]),
            "injury": float(row["injury"]),
            "sample_id": row.get("sample_id"),
            "subject_id": row.get("subject_id"),
            "study_id": row.get("study_id"),
        }


def select_genes_train_only(X: np.ndarray, metadata: pd.DataFrame, n_genes: int) -> np.ndarray:
    """Return gene indices using training observations only.

    This function refuses to select features from a hierarchy that already has
    partition leakage. Test/validation observations never enter the variance
    calculation. The caller is expected to freeze both the split and selected
    indices as part of the benchmark record.
    """
    if "_split" not in metadata:
        raise ValueError("metadata must contain '_split'")
    if X.ndim != 2 or X.shape[0] != len(metadata):
        raise ValueError("X and metadata have incompatible shapes")

    # Protect against the common failure mode where feature selection is run
    # after an accidental subject/sample split leak has already occurred.
    assert_no_leakage(metadata, split_column="_split")

    train = metadata["_split"].eq("train").to_numpy()
    if not train.any():
        raise ValueError("no training observations")
    if n_genes < 1 or n_genes > X.shape[1]:
        raise ValueError("n_genes outside expression feature range")
    variance = np.var(np.asarray(X, dtype=np.float64)[train], axis=0)
    selected = np.argsort(variance, kind="stable")[-n_genes:]
    return np.sort(selected)
