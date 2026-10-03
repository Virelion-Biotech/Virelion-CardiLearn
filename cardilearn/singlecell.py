"""Optional AnnData adapters for single-cell and single-nucleus cardiac datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def load_anndata(path: str | Path):
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("install the 'bio' extra to use AnnData support") from exc
    return ad.read_h5ad(path)


def obs_table(adata) -> pd.DataFrame:
    table = adata.obs.copy()
    table = table.reset_index(names="sample_id")
    if table["sample_id"].duplicated().any():
        raise ValueError("AnnData observation IDs must be unique")
    return table


def pseudobulk_counts(
    adata,
    *,
    group_column: str,
    layer: str | None = None,
) -> pd.DataFrame:
    """Aggregate counts by biological group without densifying the full matrix."""
    if group_column not in adata.obs:
        raise KeyError(f"missing AnnData obs column: {group_column}")
    matrix = adata.layers[layer] if layer else adata.X
    groups = pd.Series(adata.obs[group_column].astype(str).to_numpy(), name=group_column)
    codes, uniques = pd.factorize(groups, sort=False)
    if hasattr(matrix, "tocsr"):
        sparse_matrix = matrix.tocsr()
        indicator = sparse.csr_matrix(
            (
                np.ones(len(codes), dtype=np.float32),
                (codes, np.arange(len(codes))),
            ),
            shape=(len(uniques), sparse_matrix.shape[0]),
        )
        aggregated = (indicator @ sparse_matrix).toarray()
    else:
        dense = np.asarray(matrix)
        aggregated = np.zeros((len(uniques), dense.shape[1]), dtype=np.float64)
        for code in range(len(uniques)):
            aggregated[code] = dense[codes == code].sum(axis=0)
    return pd.DataFrame(aggregated, index=uniques.astype(str), columns=adata.var_names).reset_index(names=group_column)
