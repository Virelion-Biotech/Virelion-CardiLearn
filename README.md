# Virelion CardiLearn

**A research-grade cardiac state representation model with explicit biological validation boundaries.**

CardiLearn is the learning layer of the Virelion cardiac ML ecosystem. `main` now tracks the new CardiLearn research-model direction: a structured transcriptomic representation model that learns from gene expression, compresses expression into learned molecular programs, conditions representations on assay/species context, separates shared and private latent information, and exposes biological prediction heads.

> **Current status:** CardiLearn v0.1 is an executable research prototype. It is not a validated cardiac foundation model, a clinical system, or evidence of regenerative efficacy. Real-data scientific claims remain data-dependent and must be demonstrated on locked held-out benchmarks.

## Current model direction

The core learning object is a shared cardiac cell-state representation:

```text
expression
   -> gene tokens
   -> learned molecular program cross-attention
   -> molecular representation
   + species/assay context
   -> contextual modulation
   -> shared/private latent state
   -> maturation / injury / cell-state heads
   -> reconstruction
```

The prototype intentionally begins small enough for CPU/12-GB RAM experiments:

- 2,000 input genes
- 64-dimensional gene tokens
- 16 learned molecular program queries
- 128-dimensional shared latent
- 32-dimensional private latent

The prototype implementation is in `cardilearn/prototype/`.

## Why the repository still contains other ML code

The existing scikit-learn models, PCA embedding, preprocessing, splitting, provenance, benchmark, calibration, and explainability utilities remain because they are **baselines or supporting research infrastructure**. They are not the new CardiLearn model.

The obsolete standalone PyTorch MLP implementation previously exposed through `cardilearn/deep.py` has been removed from `main`.

## Scientific formulation

CardiLearn is being developed around a multi-objective representation-learning problem. The initial core objectives are:

- molecular reconstruction;
- technical-view invariance;
- anti-collapse regularization;
- biological-state supervision;
- maturation ordering/regression;
- injury-state prediction.

Regeneration is intentionally **not** represented as a hand-written marker score. The planned regeneration objective is a sample-level relational/ranking objective derived from evidence-backed biological contrasts. Cross-species alignment and population-state dynamics are later modules that will only be activated after the core representation passes its benchmarks.

## Data hierarchy and leakage controls

The canonical biological hierarchy is:

```text
study -> subject -> sample -> cell
```

Cells are learning observations, but biological evidence and primary inferential comparisons must respect independent biological units. Whole studies remain intact for primary train/validation/test partitions. Train-derived feature selection and preprocessing must never inspect held-out observations.

For single-cell and single-nucleus data, CardiLearn is designed to learn a shared representation while retaining assay-aware context rather than blindly treating scRNA-seq and snRNA-seq as identical measurements.

## Baseline strategy

The first mandatory comparison is deliberately small:

1. PCA + linear probes
2. plain MLP
3. plain autoencoder
4. CardiLearn v0.1

The structured model only earns additional complexity if it improves held-out biological generalization and survives shortcut/robustness tests.

## Validation priorities

Primary evaluation is intended to include:

- unseen-study prediction;
- subject/biological-replicate integrity;
- technical perturbation robustness;
- study-ID shortcut probes;
- modality/species confounding checks;
- independent external validation when available.

Pretty embeddings or reconstruction loss alone do not establish biological validity.

## Repository structure

```text
cardilearn/
  prototype/           # New CardiLearn v0.1 research model
    model.py           # Gene-token + learned-program + shared/private architecture
    losses.py          # Prototype training objectives
  adapters.py          # Data adapters
  benchmark_matrix.py  # Candidate-model matrix
  benchmark_runner.py  # Benchmark execution
  benchmarks.py        # Leakage-safe benchmark utilities
  calibration.py       # Calibration/uncertainty metrics
  cardiobench.py       # CardiBench definitions and holdouts
  data.py              # Dataset contracts
  embeddings.py        # PCA representation baseline
  neural.py            # Scikit-learn MLP baseline
  provenance.py        # Run/data provenance
  splitting.py         # Leakage-safe splitting
  training.py          # Existing training orchestration
  validation.py        # Dataset integrity checks
  ...

docs/
  CARDILEARN_MODEL_V0_1.md
  ...
configs/
scripts/
tests/
```

## Migration note

As the new model is developed, `main` is the authoritative branch. New CardiLearn architecture, losses, experiments, and documentation should extend the v0.1 research direction rather than resurrecting the removed standalone deep MLP as the primary model.

## Data and benchmark policy

Source datasets are not committed to Git. Raw data remain externally sourced and fingerprinted. Benchmark performance is not claimed until the corresponding data have been materialized and the evaluation is independently reproduced under the declared split policy.

## Ecosystem

- **CardiAgent** — cardiac challenge-agent generation
- **CardiVex** — challenge detection and characterization
- **CardiBench** — curated benchmarks and canonical splits
- **CardiLearn** — cardiac representation learning and model training
- **CardiEval** — independent evaluation and statistical comparison
- **CardiAtlas** — cardiac literature and omics/phenotype knowledge base
- **CardiSim** — synthetic cardiac trajectory simulation
- **CardiTrace** — provenance and reproducibility
- **CardiBridge** — cross-component schemas and APIs

## License

MIT
