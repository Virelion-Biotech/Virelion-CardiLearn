"""Optional AnnData adapters for single-cell and single-nucleus cardiac datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_anndata(path: str | Path):
    """Load an .h5ad file through the optional bioinformatics extra."""
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("install the 'bio' extra to use AnnData support") from exc
    return ad.read_h5ad(path)


def obs_table(adata) -> pd.DataFrame:
    """Return observation metadata with a stable sample_id column."""
    table = adata.obs.copy()
    table = table.reset_index(names="sample_id")
    if table["sample_id"].duplicated().any():
        raise ValueError("AnnData observation IDs must be unique")
    return table


def pseudobulk_counts(adata, *, group_column: str, layer: str | None = None) -> pd.DataFrame:
    """Aggregate expression counts by biological group, preserving group identity."""
    if group_column not in adata.obs:
        raise KeyError(f"missing AnnData obs column: {group_column}")
    matrix = adata.layers[layer] if layer else adata.X
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    frame = pd.DataFrame(matrix, columns=adata.var_names, index=adata.obs.index)
    frame[group_column] = adata.obs[group_column].to_numpy()
    return frame.groupby(group_column, sort=False).sum(numeric_only=True).reset_index()
