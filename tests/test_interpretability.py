from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from cardilearn.interpretability import (
    aggregate_ortholog_attribution,
    attribute_latent,
    gene_mask_counterfactual,
    gsea_enrichment_score,
    overrepresentation_enrichment,
    permutation_importance,
    rank_genes,
)
from cardilearn.prototype.model import CardiLearnProto


def test_integrated_gradients_latent_and_gene_ranking():
    torch.manual_seed(7)
    model = CardiLearnProto(
        n_genes=8, n_species=2, n_assays=2, n_cell_types=2,
        gene_dim=8, n_programs=2, shared_dim=4, private_dim=2,
    )
    model.eval()
    x = torch.randn(4, 8)
    species = torch.zeros(4, dtype=torch.long)
    assay = torch.zeros(4, dtype=torch.long)
    attribution = attribute_latent(model, x, species, assay, latent_index=1, steps=4)
    assert attribution.shape == x.shape
    ranked = rank_genes(attribution, [f"G{i}" for i in range(8)])
    assert list(ranked.columns) == ["gene", "score"]
    assert len(ranked) == 8


def test_masking_and_permutation_are_deterministic():
    torch.manual_seed(1)
    x = torch.randn(6, 5)

    def score(values: torch.Tensor) -> torch.Tensor:
        return values[:, 0] - 0.5 * values[:, 1]

    masked = gene_mask_counterfactual(score, x, [0, 1])
    assert masked.loc[0, "n_masked_features"] == 2
    first = permutation_importance(score, x, repeats=3, seed=11)
    second = permutation_importance(score, x, repeats=3, seed=11)
    assert np.allclose(first["importance_mean"], second["importance_mean"])


def test_enrichment_and_fdr():
    ranked = ["A", "B", "C", "D", "E", "F"]
    sets = {"pathway_1": {"A", "B", "Z"}, "pathway_2": {"E", "F"}}
    result = overrepresentation_enrichment(ranked, sets, top_k=3)
    assert result.iloc[0]["gene_set"] == "pathway_1"
    assert (result["fdr"] >= result["p_value"]).all()
    score, trace = gsea_enrichment_score(ranked, {"A", "B"})
    assert score > 0
    assert len(trace) == len(ranked)


def test_ortholog_aggregation():
    result = aggregate_ortholog_attribution(
        {"M1": 0.5, "M2": 0.7, "H1": 0.2},
        {"M1": "OG1", "M2": "OG2", "H1": "OG1"},
    )
    assert set(result["ortholog_group"]) == {"OG1", "OG2"}
    assert result.loc[result["ortholog_group"] == "OG1", "n_genes"].iloc[0] == 2
