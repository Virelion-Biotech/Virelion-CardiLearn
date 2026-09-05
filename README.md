# Virelion CardiLearn

**A reproducible research prototype for cardiac molecular-state representation learning.**

CardiLearn is the learning layer of the Virelion cardiac ML ecosystem. This repository contains a deliberately small reference architecture for transcriptomic representation learning, together with reproducible data, validation, interpretation, perturbation, and benchmarking infrastructure.

> **Scientific status:** CardiLearn is an executable research prototype, not a validated cardiac foundation model, clinical system, or evidence of regenerative efficacy. The current neural architecture is intentionally CPU-friendly (2,000 input genes, 64-dimensional gene tokens, 16 learned program queries). Its purpose is to validate the data contracts, leakage controls, objectives, evaluation protocol, and end-to-end research workflow before scaling model capacity. Real-data scientific claims must be demonstrated on locked held-out benchmarks and, where appropriate, independent biological validation.

## Scope: prototype first, claims later

The project intentionally separates **software maturity** from **scientific maturity**. A sophisticated validation framework does not make a small prototype biologically validated. CardiLearn therefore uses the following order:

```text
software correctness
   ↓
data / metadata audit
   ↓
locked real-data cohort
   ↓
training
   ↓
locked benchmark
   ↓
unseen-study / species validation
   ↓
interpretation
   ↓
independent biological validation
```

Until the real-data lock and experiments are executed, the repository should be read as **research infrastructure plus a model prototype**, not as a completed biological result.

## Research model

```text
expression
   → gene tokens
   → learned molecular programs
   → molecular representation
   + species / assay context
   → shared + private latent state
   → maturation / injury / cell-state heads
   → reconstruction
```

The current reference architecture is intentionally small:

- 2,000 input genes
- 64-dimensional gene tokens
- 16 learned molecular program queries
- 128-dimensional shared latent
- 32-dimensional private latent

These dimensions are **prototype settings, not claims of biological sufficiency**. Scaling the architecture is a later experimental variable and must earn its complexity through held-out evidence.

Implementation: `cardilearn/prototype/`.

## Research capabilities

| Layer | Purpose |
|---|---|
| Data + validation | explicit contracts, provenance, integrity checks, leakage-safe splits |
| Representation | shared/private cardiac molecular-state representation |
| Interpretability | attribution, masking/permutation, enrichment, counterfactuals, ortholog conservation |
| Perturbation | predictive latent-response modeling with uncertainty |
| Benchmarking | matched baseline comparisons with locked evaluation rules |

Detailed architecture: `docs/ARCHITECTURE.md`.

Research usage guide: `docs/USER_GUIDE_V0_3.md`.

Documentation standard: `docs/STEP17_DOCUMENTATION.md`.

## Scientific formulation

The initial learning objectives include molecular reconstruction, technical-view invariance, anti-collapse regularization, biological-state supervision, maturation ordering/regression, and injury-state prediction. Regeneration is not implemented as a hand-written marker score. Cross-species alignment and regeneration/transition objectives require evidence-backed datasets and their own locked evaluation protocols.

## Data hierarchy and leakage controls

```text
study family
  ↓
study
  ↓
subject / donor / animal
  ↓
sample / library
  ↓
cell / nucleus
```

Cells are observations, not automatically independent biological replicates. Related biological units must not cross declared train/validation/test boundaries. Train-derived preprocessing, feature selection, dimensionality reduction, and imputation must not inspect held-out observations.

## Real-data pilot

The metadata-first pilot currently includes complementary candidate studies:

- **GSE185289** — pig single-nucleus data spanning fetal/postnatal, injury, and regenerative/non-regenerative contexts.
- **GSE153480** — neonatal mouse single-cell data spanning P1/P8 MI and sham conditions.
- **GSE217494** — human CITE-seq/GEX data spanning healthy and ischemic cardiomyopathy/MI contexts.

These are **candidate studies, not a locked benchmark**. Subject/sample relationships, technical replicates, conditions, regions, timepoints, assay boundaries, and study-family independence must be reconciled before a scientific data lock.

Expression matrices are not committed to Git.

## Baseline strategy

The mandatory initial comparison is:

1. PCA + linear probe
2. plain MLP
3. plain autoencoder
4. CardiLearn

The structured model only earns additional complexity if it improves held-out biological generalization and survives shortcut/robustness tests.

## Interpretability

Step 13 provides multiple complementary analyses rather than treating attention as explanation:

- Integrated Gradients;
- masking and permutation sensitivity;
- latent/program analysis;
- gene-set enrichment;
- counterfactual masking;
- explicit ortholog-group conservation.

These are interpretation aids, not causal evidence.

## Perturbation prediction

Step 14 predicts latent responses from a baseline state plus perturbation identity/type, dose, and duration. It is deliberately documented as predictive infrastructure. A predicted response is not a causal effect without independent experimental validation.

## Benchmarking

Step 15 defines a reproducible comparison protocol with fixed dataset/task/split semantics, repeated seeds, model manifests, fingerprints, and paired statistical comparisons. Test data remain locked until model selection is complete.

## CLI

```bash
pip install -e .
cardilearn models --task classification
cardilearn validate --data data.csv --target label
cardilearn train --data data.csv --target label --output runs/example
cardilearn benchmark-info --definition configs/benchmark_v1.yaml
```

Optional dependencies are separated in `pyproject.toml`; PyTorch is not required for metadata/validation infrastructure because prototype imports are lazy.

## Repository structure

```text
cardilearn/
├── prototype/            # small reference representation-learning model
├── interpretability/     # Step 13
├── perturbation/         # Step 14
├── benchmark_protocol.py # Step 15 evaluation contract
├── benchmark_suite.py    # Step 15 baseline implementations
├── real_data.py          # real-data contracts/parser
├── data.py               # dataset contracts
├── splitting.py          # leakage-safe splitting
├── validation.py         # dataset integrity checks
├── provenance.py         # run/data provenance
└── ...

docs/
configs/
scripts/
tests/
```

## Naming convention

**Cardi-** is the Virelion software/platform family prefix (for example, CardiLearn, CardiBench, CardiEval, CardiAtlas, CardiSim, CardiTrace, and CardiBridge). Functional measurement or modality tools may retain descriptive names such as **ElectroTrace**, **MyoTrace**, and **OptiCell**. This distinction is intentional: `Cardi-` identifies the software family, while modality names identify what is being measured.

## Scientific maturity gate

The maturity gate is a governance protocol, not evidence that the model has already passed it. Each stage requires actual evidence before the next claim is made:

```text
software tests
   ↓
data/metadata audit
   ↓
data lock
   ↓
training
   ↓
locked benchmark
   ↓
unseen-study/species validation
   ↓
interpretation
   ↓
independent biological validation
```

A passing CI build demonstrates software correctness, not biological validity. No benchmark or biological performance claim is made until real datasets are materialized and the declared protocol is executed.

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

## Current priority

The next meaningful milestone is **not another architectural feature**. It is the first complete real-data result: a reviewed cohort, frozen split, train-only preprocessing, reproducible training run, locked benchmark against the required baselines, and held-out evaluation. Until that exists, model capacity and additional ecosystem surface area remain secondary.

## License

MIT
