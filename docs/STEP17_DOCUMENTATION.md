# Step 17 — Documentation Standard

## Purpose

Step 17 makes CardiLearn understandable and reproducible by a researcher who did not build the repository.

## Scientific identity

CardiLearn is a research representation-learning system for cardiac molecular state. Its intended scientific target is reusable representation of biological state across development, maturation, injury, regeneration, and species. The current repository contains a research prototype and supporting infrastructure; it does not yet establish that the intended representation has been learned successfully on real-world data.

## What a valid experiment must specify

Every reported experiment should identify:

- dataset and source version;
- study/subject/sample hierarchy;
- inclusion and exclusion criteria;
- feature universe and gene identifiers;
- modality and assay;
- train/validation/test assignment policy;
- frozen split manifest or equivalent fingerprint;
- preprocessing and which partition fitted each transform;
- model architecture and hyperparameters;
- loss/objective configuration;
- random seeds;
- hardware/software environment;
- evaluation metrics;
- model-selection rule;
- analysis partition used for interpretation.

## Scientific claim boundaries

The repository uses the following distinctions:

| Output | What it supports | What it does not prove |
|---|---|---|
| latent separation | representation differs between groups | biological mechanism |
| attribution | model relies on a feature | feature causes phenotype |
| pathway enrichment | ranked genes overlap a biological set | pathway is causally responsible |
| perturbation prediction | model predicts a response pattern | intervention causes that response |
| cross-species concordance | similar mapped program evidence | species are biologically identical |
| benchmark improvement | better performance under a declared protocol | universal superiority |

## Data leakage rules

The strongest default split boundary is the study/biological unit. Related observations must not cross a held-out boundary. For single-cell and single-nucleus datasets, cells from one sample are not independent biological replicates.

Any train-derived transformation—including feature selection, normalization parameters, imputation, dimensionality reduction, or learned gene sets—must be fit without access to held-out observations.

## Interpretation rules

Do not use attention weights as the sole explanation. Prefer convergent evidence from attribution, masking/permutation sensitivity, enrichment, counterfactual analysis, and cross-species ortholog mapping.

Interpretability performed on a test set must remain descriptive. If interpretation changes the model, features, thresholds, or training procedure, that constitutes a new experiment and requires a fresh held-out evaluation.

## Benchmark rules

Benchmark comparisons must keep the dataset, split, task definition, and evaluation protocol fixed. Hyperparameter selection belongs to training/validation data. The test set is evaluated only after the model is frozen.

Repeated seeds should be reported where computationally practical. Statistical comparisons should respect the paired nature of repeated runs and should not be used to manufacture significance from non-independent cells.

## Reproducibility artifact

A complete run should be exportable as a bundle containing, at minimum:

```text
run/
├── config
├── dataset fingerprint
├── feature manifest
├── split manifest
├── model artifact
├── metrics
├── predictions
└── provenance
```

The artifact must be sufficient to determine exactly what was trained and evaluated, even if the original raw dataset is no longer stored locally.

## Real-data maturity gate

The project should not transition from prototype claims to scientific conclusions merely because code runs. The minimum progression is:

```text
prototype passes software tests
        ↓
metadata and data audit
        ↓
data lock
        ↓
training runs
        ↓
locked benchmark
        ↓
unseen-study/species validation
        ↓
interpretability
        ↓
independent biological validation
```

## Documentation maintenance

When a public API, model objective, data contract, benchmark protocol, or scientific assumption changes, update the corresponding documentation in the same change set. Documentation should describe implemented behavior, not planned behavior presented as if it already exists.
