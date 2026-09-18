"""Conserved-gene and cross-species representation primitives."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class GeneGroupMap:
    """Validated mapping from model genes to conserved functional groups."""

    gene_to_group: tuple[int, ...]
    n_groups: int

    def __post_init__(self) -> None:
        if not self.gene_to_group or self.n_groups < 1:
            raise ValueError("gene_to_group and n_groups must be non-empty/positive")
        values = np.asarray(self.gene_to_group, dtype=int)
        if values.min() < 0 or values.max() >= self.n_groups:
            raise ValueError("gene group IDs must lie in [0, n_groups)")
    
    @property
    def n_genes(self) -> int:
        return len(self.gene_to_group)


def validate_gene_group_map(group_ids, n_genes: int, n_groups: int) -> np.ndarray:
    values = np.asarray(group_ids, dtype=int)
    if values.ndim != 1 or len(values) != n_genes:
        raise ValueError("group_ids must be [n_genes]")
    if n_groups < 1 or values.size == 0:
        raise ValueError("n_groups must be positive and group_ids non-empty")
    if np.any(values < 0) or np.any(values >= n_groups):
        raise ValueError("group IDs outside declared group range")
    return values


class ConservedGeneIdentity(nn.Module):
    """Shared functional-group embedding plus gene-specific residual."""

    def __init__(
        self,
        n_genes: int,
        dim: int,
        group_ids,
        n_groups: int,
        *,
        residual_scale: float = 0.02,
    ) -> None:
        super().__init__()
        values = validate_gene_group_map(group_ids, n_genes, n_groups)
        self.n_genes = n_genes
        self.dim = dim
        self.n_groups = n_groups
        self.conserved_embedding = nn.Embedding(n_groups, dim)
        self.gene_residual = nn.Parameter(torch.empty(n_genes, dim))
        self.register_buffer("group_ids", torch.as_tensor(values, dtype=torch.long), persistent=True)
        nn.init.normal_(self.conserved_embedding.weight, mean=0.0, std=residual_scale)
        nn.init.normal_(self.gene_residual, mean=0.0, std=residual_scale)

    def forward(self) -> torch.Tensor:
        return self.conserved_embedding(self.group_ids) + self.gene_residual


def functional_group_pool(
    gene_states: torch.Tensor,
    group_ids: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    """Mean-pool gene states into conserved functional groups."""
    if gene_states.ndim != 3:
        raise ValueError("gene_states must be [batch, genes, dim]")
    if group_ids.ndim != 1 or group_ids.shape[0] != gene_states.shape[1]:
        raise ValueError("group_ids must have one entry per gene")
    if torch.any(group_ids < 0) or torch.any(group_ids >= n_groups):
        raise ValueError("group IDs outside declared group range")
    one_hot = torch.nn.functional.one_hot(group_ids.long(), num_classes=n_groups).to(gene_states.dtype)
    pooled = torch.einsum("bgd,gk->bkd", gene_states, one_hot)
    counts = one_hot.sum(dim=0).clamp_min(1.0).to(gene_states.dtype)
    return pooled / counts.view(1, -1, 1)
