"""Ortholog-aware aggregation for cross-species interpretation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def aggregate_ortholog_attribution(
    gene_scores: Mapping[str, float],
    ortholog_map: Mapping[str, str],
) -> pd.DataFrame:
    """Aggregate gene-level scores into explicit ortholog groups."""
    rows: dict[str, list[float]] = {}
    for gene, score in gene_scores.items():
        group = ortholog_map.get(gene)
        if group is None:
            continue
        rows.setdefault(group, []).append(float(score))
    output = []
    for group, scores in rows.items():
        output.append(
            {
                "ortholog_group": group,
                "n_genes": len(scores),
                "mean_score": float(np.mean(scores)),
                "max_abs_score": float(np.max(np.abs(scores))),
                "signed_sum": float(np.sum(scores)),
            }
        )
    return pd.DataFrame(output).sort_values("mean_score", ascending=False).reset_index(drop=True) if output else pd.DataFrame(
        columns=["ortholog_group", "n_genes", "mean_score", "max_abs_score", "signed_sum"]
    )


def conserved_program_score(
    species_gene_scores: Mapping[str, Mapping[str, float]],
    ortholog_maps: Mapping[str, Mapping[str, str]],
    *,
    min_species: int = 2,
) -> pd.DataFrame:
    """Find ortholog groups supported across multiple species.

    Each species contributes the mean signed attribution of its mapped genes. The
    result distinguishes conservation of direction from conservation of magnitude.
    """
    if min_species < 1:
        raise ValueError("min_species must be positive")
    combined: dict[str, list[float]] = {}
    support: dict[str, set[str]] = {}
    for species, scores in species_gene_scores.items():
        mapping = ortholog_maps.get(species, {})
        grouped: dict[str, list[float]] = {}
        for gene, score in scores.items():
            group = mapping.get(gene)
            if group is not None:
                grouped.setdefault(group, []).append(float(score))
        for group, values in grouped.items():
            combined.setdefault(group, []).append(float(np.mean(values)))
            support.setdefault(group, set()).add(species)

    rows = []
    for group, values in combined.items():
        species = sorted(support[group])
        if len(species) < min_species:
            continue
        signed = np.asarray(values, dtype=float)
        rows.append(
            {
                "ortholog_group": group,
                "n_species": len(species),
                "species": ",".join(species),
                "mean_score": float(signed.mean()),
                "mean_abs_score": float(np.abs(signed).mean()),
                "direction_concordance": float(abs(signed.mean()) / (np.abs(signed).mean() + 1e-12)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["ortholog_group", "n_species", "species", "mean_score", "mean_abs_score", "direction_concordance"]
    )


def conserved_gene_intersection(
    gene_lists: Sequence[Sequence[str]],
    ortholog_map: Mapping[str, str],
) -> set[str]:
    """Return ortholog groups represented in every supplied species list."""
    if not gene_lists:
        return set()
    mapped_sets = [{ortholog_map[g] for g in genes if g in ortholog_map} for genes in gene_lists]
    return set.intersection(*mapped_sets) if mapped_sets else set()
