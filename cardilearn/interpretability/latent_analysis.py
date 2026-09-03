"""Utilities for interpreting CardiLearn latent dimensions and programs."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from .attribution import rank_genes


def summarize_latent_dimension(
    attributions: torch.Tensor | np.ndarray,
    *,
    latent_index: int,
    gene_names: Sequence[str] | None = None,
    top_k: int = 25,
) -> pd.DataFrame:
    """Return the strongest input genes associated with one latent dimension."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ranked = rank_genes(attributions, gene_names, aggregate="mean_signed")
    result = ranked.assign(latent_index=latent_index)
    return result.head(top_k).reset_index(drop=True)


def latent_gene_matrix(
    latent_attributions: dict[int, torch.Tensor | np.ndarray],
    gene_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create a latent-by-gene signed attribution matrix for downstream analysis."""
    if not latent_attributions:
        raise ValueError("latent_attributions cannot be empty")
    vectors = []
    indices = []
    for latent_index in sorted(latent_attributions):
        value = np.asarray(
            latent_attributions[latent_index].detach().cpu()
            if torch.is_tensor(latent_attributions[latent_index])
            else latent_attributions[latent_index]
        )
        if value.ndim == 2:
            value = value.mean(axis=0)
        if value.ndim != 1:
            raise ValueError("each latent attribution must have shape [samples, genes] or [genes]")
        vectors.append(value)
        indices.append(latent_index)
    matrix = np.stack(vectors, axis=0)
    names = list(gene_names) if gene_names is not None else [f"gene_{i}" for i in range(matrix.shape[1])]
    if len(names) != matrix.shape[1]:
        raise ValueError("gene_names length must match the number of genes")
    return pd.DataFrame(matrix, index=indices, columns=names)


def program_gene_weights(
    program_attention: torch.Tensor,
    *,
    top_k: int = 25,
    gene_names: Sequence[str] | None = None,
) -> dict[int, pd.DataFrame]:
    """Summarize program-to-gene attention as a discovery aid.

    Attention is reported explicitly as attention-derived evidence; it should be
    corroborated with gradient or perturbation attribution before calling a gene
    mechanistically important.
    """
    if program_attention.ndim == 4:
        # [batch, heads, programs, genes]
        weights = program_attention.mean(dim=(0, 1))
    elif program_attention.ndim == 3:
        # [batch, programs, genes]
        weights = program_attention.mean(dim=0)
    else:
        raise ValueError("program_attention must have shape [B,H,K,G] or [B,K,G]")
    weights_np = weights.detach().cpu().numpy()
    names = list(gene_names) if gene_names is not None else [f"gene_{i}" for i in range(weights_np.shape[1])]
    if len(names) != weights_np.shape[1]:
        raise ValueError("gene_names length must match the number of genes")
    output: dict[int, pd.DataFrame] = {}
    for program_index, row in enumerate(weights_np):
        order = np.argsort(-row)[:top_k]
        output[program_index] = pd.DataFrame(
            {"gene": [names[i] for i in order], "attention": row[order]}
        ).reset_index(drop=True)
    return output


def latent_similarity_matrix(z: torch.Tensor | np.ndarray, *, metric: str = "cosine") -> np.ndarray:
    """Compute a pairwise latent similarity/distance matrix for state structure."""
    values = np.asarray(z.detach().cpu() if torch.is_tensor(z) else z, dtype=float)
    if values.ndim != 2:
        raise ValueError("z must have shape [samples, dimensions]")
    if metric == "cosine":
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        normalized = values / np.clip(norms, 1e-12, None)
        return normalized @ normalized.T
    if metric == "euclidean":
        diffs = values[:, None, :] - values[None, :, :]
        return np.sqrt(np.sum(diffs * diffs, axis=-1))
    raise ValueError("metric must be 'cosine' or 'euclidean'")
