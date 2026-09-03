"""Interpretability API for CardiLearn.

These utilities are explanation aids, not causal inference. Any mechanistic claim
requires independent biological validation.
"""
from .attribution import (
    attribute_latent,
    attribute_prediction,
    integrated_gradients,
    mask_feature_sensitivity,
    permutation_importance,
    rank_genes,
)
from .counterfactual import gene_mask_counterfactual, ranked_gene_counterfactuals
from .enrichment import gsea_enrichment_score, overrepresentation_enrichment
from .latent_analysis import latent_gene_matrix, latent_similarity_matrix, program_gene_weights, summarize_latent_dimension
from .ortholog_conservation import aggregate_ortholog_attribution, conserved_gene_intersection, conserved_program_score

__all__ = [
    "aggregate_ortholog_attribution",
    "attribute_latent",
    "attribute_prediction",
    "conserved_gene_intersection",
    "conserved_program_score",
    "gene_mask_counterfactual",
    "gsea_enrichment_score",
    "integrated_gradients",
    "latent_gene_matrix",
    "latent_similarity_matrix",
    "mask_feature_sensitivity",
    "overrepresentation_enrichment",
    "permutation_importance",
    "program_gene_weights",
    "rank_genes",
    "ranked_gene_counterfactuals",
    "summarize_latent_dimension",
]
