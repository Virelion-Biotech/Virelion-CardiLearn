"""Scalable ingestion primitives for curated single-cell and single-nucleus data.

The ingestion layer is deliberately conservative: it preserves source identifiers,
keeps expression sparse until a train-ready matrix is requested, and refuses to
invent biological labels. It is a preparation layer, not a data-lock decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


REQUIRED_METADATA = (
    "study_id",
    "subject_id",
    "sample_id",
    "species",
    "assay",
    "cell_type",
    "maturation",
    "injury",
)


@dataclass(frozen=True)
class SparseExpression:
    """Expression matrix with observations in rows and genes in columns."""

    X: sparse.csr_matrix
    observation_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.X.ndim != 2:
            raise ValueError("expression matrix must be two-dimensional")
        if self.X.shape != (len(self.observation_ids), len(self.gene_ids)):
            raise ValueError("expression dimensions do not match observation/gene IDs")
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("observation IDs must be unique within an expression matrix")
        if len(set(self.gene_ids)) != len(self.gene_ids):
            raise ValueError("gene IDs must be unique after gene reconciliation")
        if not np.isfinite(self.X.data).all():
            raise ValueError("expression contains non-finite values")


def read_10x_mtx(matrix_dir: str | Path, *, make_csr: bool = True) -> SparseExpression:
    """Read a 10x Matrix Market directory without densifying expression.

    Both modern ``features.tsv`` and legacy ``genes.tsv`` layouts are accepted.
    The latter is common in older GEO 10x deposits, so acquisition does not need
    to rename or mutate source files before ingestion.
    """
    root = Path(matrix_dir)
    matrix_path = root / "matrix.mtx"
    if not matrix_path.exists():
        matrix_path = root / "matrix.mtx.gz"
    if not matrix_path.exists():
        raise FileNotFoundError("10x matrix.mtx(.gz) not found")

    barcode_path = root / "barcodes.tsv"
    if not barcode_path.exists():
        barcode_path = root / "barcodes.tsv.gz"
    feature_path = root / "features.tsv"
    if not feature_path.exists():
        feature_path = root / "features.tsv.gz"
    if not feature_path.exists():
        feature_path = root / "genes.tsv"
    if not feature_path.exists():
        feature_path = root / "genes.tsv.gz"
    if not barcode_path.exists() or not feature_path.exists():
        raise FileNotFoundError(
            "10x barcodes.tsv(.gz) and features.tsv(.gz) or genes.tsv(.gz) are required"
        )

    matrix = mmread(matrix_path)
    matrix = matrix.tocsr() if make_csr else sparse.csr_matrix(matrix)
    barcodes = pd.read_csv(barcode_path, sep="\t", header=None, compression="infer")
    features = pd.read_csv(feature_path, sep="\t", header=None, compression="infer")
    if len(features) == matrix.shape[0] and matrix.shape[0] != len(barcodes):
        matrix = matrix.T.tocsr()
    if matrix.shape != (len(barcodes), len(features)):
        raise ValueError(
            f"10x dimensions mismatch: matrix={matrix.shape}, "
            f"barcodes={len(barcodes)}, features={len(features)}"
        )
    gene_column = 1 if features.shape[1] >= 2 else 0
    genes = tuple(features.iloc[:, gene_column].astype(str))
    obs = tuple(barcodes.iloc[:, 0].astype(str))
    result = SparseExpression(matrix.astype(np.float32), obs, genes)
    result.validate()
    return result


def validate_metadata(metadata: pd.DataFrame) -> None:
    """Validate the canonical hierarchy without assigning missing biology."""
    missing = [column for column in REQUIRED_METADATA if column not in metadata.columns]
    if missing:
        raise ValueError(f"missing canonical metadata columns: {missing}")
    if metadata[list(REQUIRED_METADATA)].isna().any().any():
        raise ValueError("canonical metadata contains missing required values")
    for child, parent in (("sample_id", "subject_id"), ("subject_id", "study_id")):
        counts = metadata.groupby(child, dropna=False)[parent].nunique(dropna=False)
        if (counts > 1).any():
            bad = counts[counts > 1].index.astype(str).tolist()[:10]
            raise ValueError(f"{child} maps to multiple {parent} values: {bad}")


def apply_gene_map(
    expression: SparseExpression,
    gene_map: pd.DataFrame,
    *,
    source_column: str = "source_gene",
    target_column: str = "target_gene",
    require_one_to_one: bool = True,
) -> SparseExpression:
    """Apply an explicit gene map, retaining only unambiguous target genes."""
    for column in (source_column, target_column):
        if column not in gene_map.columns:
            raise ValueError(f"gene map missing '{column}'")
    mapping = gene_map[[source_column, target_column]].dropna().astype(str)
    if require_one_to_one:
        if mapping[source_column].duplicated().any() or mapping[target_column].duplicated().any():
            raise ValueError("one-to-one gene map required; duplicate source/target IDs found")
    lookup = dict(zip(mapping[source_column], mapping[target_column]))
    keep = [i for i, gene in enumerate(expression.gene_ids) if gene in lookup]
    if not keep:
        raise ValueError("gene map has no overlap with expression gene IDs")
    targets = [lookup[expression.gene_ids[i]] for i in keep]
    if len(set(targets)) != len(targets):
        raise ValueError("gene mapping creates duplicate target genes")
    X = expression.X[:, keep].tocsr()
    result = SparseExpression(X, expression.observation_ids, tuple(targets))
    result.validate()
    return result


def select_variable_genes_sparse(
    expression: SparseExpression,
    observation_mask: Iterable[bool],
    n_genes: int,
) -> np.ndarray:
    """Select highest-variance genes using only the supplied observations.

    Computation remains sparse and only the selected columns are returned. This
    avoids converting a large single-cell matrix to a dense array during the
    data-lock/preprocessing stage.
    """
    expression.validate()
    mask = np.asarray(list(observation_mask), dtype=bool)
    if len(mask) != expression.X.shape[0]:
        raise ValueError("observation mask length does not match expression rows")
    if not mask.any():
        raise ValueError("observation mask contains no selected observations")
    if n_genes < 1 or n_genes > expression.X.shape[1]:
        raise ValueError("n_genes outside expression feature range")
    X = expression.X[mask]
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    order = np.argsort(variance, kind="stable")[-n_genes:]
    return np.sort(order)
