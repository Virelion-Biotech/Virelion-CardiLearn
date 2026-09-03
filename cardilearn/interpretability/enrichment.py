"""Dependency-light gene-set enrichment for CardiLearn latent programs."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import pandas as pd


def _hypergeom_tail(k: int, K: int, n: int, N: int) -> float:
    """P(X >= k) for X~Hypergeometric(N, K, n)."""
    if not 0 <= k <= min(K, n):
        return 0.0 if k > min(K, n) else 1.0
    lo = max(0, n - (N - K))
    k = max(k, lo)
    den = math.comb(N, n)
    return min(
        1.0,
        sum(math.comb(K, i) * math.comb(N - K, n - i) for i in range(k, min(K, n) + 1)) / den,
    )


def overrepresentation_enrichment(
    ranked_genes: Sequence[str],
    gene_sets: dict[str, Iterable[str]],
    *,
    top_k: int | None = None,
    background_genes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run exact hypergeometric over-representation analysis.

    ``ranked_genes`` should be ordered from most positive/important to least.
    The default foreground is the complete ranked list; ``top_k`` creates a
    conventional top-gene foreground. Gene identifiers are treated as strings.
    """
    ordered = list(dict.fromkeys(ranked_genes))
    if not ordered:
        raise ValueError("ranked_genes cannot be empty")
    k = len(ordered) if top_k is None else top_k
    if k < 1 or k > len(ordered):
        raise ValueError("top_k must be between 1 and the number of unique ranked genes")
    foreground = set(ordered[:k])
    background = set(ordered) if background_genes is None else set(background_genes)
    background |= foreground
    rows = []
    for name, members in gene_sets.items():
        members_set = set(members) & background
        if not members_set:
            continue
        overlap = len(foreground & members_set)
        p = _hypergeom_tail(overlap, len(members_set), len(foreground), len(background))
        rows.append(
            {
                "gene_set": name,
                "overlap": overlap,
                "gene_set_size": len(members_set),
                "foreground_size": len(foreground),
                "background_size": len(background),
                "p_value": p,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=["gene_set", "overlap", "gene_set_size", "foreground_size", "background_size", "p_value", "fdr"]
        )
    result["fdr"] = _benjamini_hochberg(result["p_value"].tolist())
    return result.sort_values(["fdr", "p_value", "gene_set"]).reset_index(drop=True)


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=p_values.__getitem__)
    adjusted = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, p_values[idx] * m / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted


def gsea_enrichment_score(
    ranked_genes: Sequence[str],
    gene_set: Iterable[str],
) -> tuple[float, list[float]]:
    """Calculate a simple weighted-free GSEA running enrichment score."""
    ordered = list(dict.fromkeys(ranked_genes))
    target = set(gene_set)
    hits = [gene in target for gene in ordered]
    n_hits = sum(hits)
    if n_hits == 0:
        raise ValueError("gene_set has no overlap with ranked_genes")
    miss = len(ordered) - n_hits
    if miss == 0:
        miss_penalty = 0.0
    else:
        miss_penalty = 1.0 / miss
    hit_increment = 1.0 / n_hits
    running = 0.0
    trace = []
    for hit in hits:
        running += hit_increment if hit else -miss_penalty
        trace.append(running)
    score = max(trace, key=abs)
    return float(score), trace
