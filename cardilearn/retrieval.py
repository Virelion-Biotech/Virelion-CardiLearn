"""Embedding retrieval with an exact NumPy path and optional FAISS acceleration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RetrievalIndex:
    embeddings: np.ndarray
    ids: tuple[str, ...]
    normalized: bool = True
    backend: str = "numpy"
    _index: object | None = None

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        ids: Sequence[str],
        *,
        backend: str = "numpy",
    ) -> "RetrievalIndex":
        values = np.asarray(embeddings, dtype=np.float32)
        identifiers = tuple(str(x) for x in ids)
        if values.ndim != 2 or values.shape[0] != len(identifiers):
            raise ValueError("embeddings must be [n, dim] with one ID per row")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("retrieval IDs must be unique")
        if not np.isfinite(values).all():
            raise ValueError("embeddings contain non-finite values")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.clip(norms, 1e-12, None)
        index = None
        resolved_backend = backend
        if backend == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise ImportError("FAISS backend requires optional 'faiss' dependency") from exc
            index = faiss.IndexFlatIP(values.shape[1])
            index.add(values)
        elif backend != "numpy":
            raise ValueError("backend must be 'numpy' or 'faiss'")
        return cls(values, identifiers, normalized=True, backend=resolved_backend, _index=index)

    def search(self, queries: np.ndarray, *, k: int = 10) -> list[list[tuple[str, float]]]:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.embeddings.shape[1]:
            raise ValueError("queries must be [n_queries, embedding_dim]")
        if k < 1:
            raise ValueError("k must be positive")
        k = min(k, len(self.ids))
        q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        if self.backend == "faiss" and self._index is not None:
            scores, indices = self._index.search(q, k)
        else:
            similarity = q @ self.embeddings.T
            candidate = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
            row = np.arange(q.shape[0])[:, None]
            scores = np.take_along_axis(similarity, candidate, axis=1)
            order = np.argsort(-scores, axis=1)
            indices = np.take_along_axis(candidate, order, axis=1)
            scores = np.take_along_axis(scores, order, axis=1)
        return [
            [(self.ids[int(index)], float(score)) for index, score in zip(row_indices, row_scores)]
            for row_indices, row_scores in zip(indices, scores)
        ]
