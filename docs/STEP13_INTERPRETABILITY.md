# Step 13 — Interpretability

## Scientific purpose

Step 13 turns a learned CardiLearn representation into testable biological hypotheses:

authoritative input genes → latent dimensions/programs → pathway enrichment → cross-species conserved programs.

These outputs are **interpretation aids, not causal evidence**. Attention weights are never treated as sufficient explanations, and a gene/pathway contribution does not establish that manipulating it will change a biological state.

## Implemented methods

### Latent-factor interpretation

`attribute_latent()` computes Integrated Gradients from expression features to one shared-latent dimension. `rank_genes()` aggregates signed or absolute attribution across samples. `program_gene_weights()` provides a separate attention-derived view of learned program-to-gene associations and labels it as supplementary evidence.

### Prediction attribution

`attribute_prediction()` accepts any scalar-per-sample forward function, making it suitable for maturity, injury, or later regeneration/transition scores without coupling Step 13 to one prediction head.

### Perturbation sensitivity

`mask_feature_sensitivity()` masks individual genes without retraining the model. `permutation_importance()` independently permutes one gene across samples and records the resulting prediction shift. These are sensitivity analyses, not model refits.

### Enrichment

`overrepresentation_enrichment()` performs exact hypergeometric over-representation analysis and Benjamini–Hochberg correction without requiring SciPy. `gsea_enrichment_score()` supplies a lightweight preranked running enrichment score for hypothesis generation.

### Counterfactual analysis

`gene_mask_counterfactual()` tests the joint effect of masking a selected gene/program proxy. `ranked_gene_counterfactuals()` progressively masks a ranked list to evaluate whether a prediction is concentrated in a small number of highly ranked features.

### Cross-species interpretation

`aggregate_ortholog_attribution()` and `conserved_program_score()` aggregate evidence through an explicit ortholog-group mapping. Conservation is reported only where the requested number of species provide mapped evidence. Direction concordance is reported separately from magnitude so a large effect driven by discordant species is not mislabeled as conserved.

## Recommended analysis sequence

1. Freeze the trained model and locked evaluation split.
2. Compute latent or prediction attribution only on the declared analysis partition.
3. Rank genes using Integrated Gradients; use masking/permutation as orthogonal checks.
4. Map ranked genes to pathway/gene-set annotations and control FDR.
5. Compare the same biological program through explicit ortholog groups across species.
6. Run counterfactual masking as a sensitivity analysis.
7. Treat stable, cross-method, cross-species signals as candidate mechanisms for independent experimental validation.

## Leakage rule

Interpretability must not be used to select genes, pathways, thresholds, or model hyperparameters using the final held-out test set. If exploratory attribution is used to redesign the model, the resulting model is a new experiment and must be re-evaluated on an untouched holdout.

## Example

```python
from cardilearn.interpretability import attribute_latent, rank_genes

attribution = attribute_latent(
    model,
    x,
    species,
    assay,
    latent_index=17,
    steps=32,
)
ranked = rank_genes(attribution, gene_names)
print(ranked.head(25))
```

## Interpretation language

Use:

> “Gene X showed high attribution to latent dimension 17, whose top-ranked genes were enriched for calcium-handling pathways.”

Do not use:

> “Gene X causes maturation.”

The latter requires perturbational and biological validation beyond model interpretability.
