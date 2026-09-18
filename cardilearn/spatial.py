"""Spatial neighborhood encoding primitives for cardiac tissue data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SpatialGraph:
    """Directed cell-neighborhood graph represented as edge_index [2, E]."""

    edge_index: torch.Tensor
    n_nodes: int

    def validate(self) -> None:
        if self.edge_index.ndim != 2 or tuple(self.edge_index.shape[:1]) != (2,):
            raise ValueError("edge_index must have shape [2, edges]")
        if self.edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must use integer indices")
        if self.n_nodes < 1:
            raise ValueError("n_nodes must be positive")
        if torch.any(self.edge_index < 0) or torch.any(self.edge_index >= self.n_nodes):
            raise ValueError("edge_index contains an out-of-range node")


class NeighborhoodAggregator(nn.Module):
    """Permutation-invariant mean aggregation over spatial neighbors."""

    def __init__(self, input_dim: int, output_dim: int | None = None) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        output_dim = output_dim or input_dim
        self.projection = nn.Identity() if output_dim == input_dim else nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape [nodes, features]")
        graph = SpatialGraph(edge_index=edge_index, n_nodes=x.shape[0])
        graph.validate()
        edge_index = edge_index.to(device=x.device)
        src, dst = edge_index.long()
        messages = self.projection(x[src])
        aggregate = torch.zeros_like(x)
        if messages.shape[1] != x.shape[1]:
            aggregate = torch.zeros(
                x.shape[0], messages.shape[1], device=x.device, dtype=x.dtype
            )
        aggregate.index_add_(0, dst, messages)
        degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        degree.index_add_(0, dst, torch.ones(dst.shape[0], device=x.device, dtype=x.dtype))
        return aggregate / degree.clamp_min(1.0).unsqueeze(-1)


def knn_graph(coordinates: np.ndarray, k: int = 8) -> torch.Tensor:
    """Build a deterministic CPU kNN graph from spatial coordinates."""
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("coordinates must be [nodes, dimensions], dimensions >= 2")
    if k < 1 or k >= len(values):
        raise ValueError("k must be in [1, n_nodes-1]")
    if not np.isfinite(values).all():
        raise ValueError("coordinates contain non-finite values")
    distances = np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(np.take_along_axis(distances, neighbors, axis=1), axis=1)
    neighbors = np.take_along_axis(neighbors, order, axis=1)
    src = np.repeat(np.arange(len(values)), k)
    dst = neighbors.reshape(-1)
    return torch.as_tensor(np.stack([src, dst], axis=0), dtype=torch.long)
