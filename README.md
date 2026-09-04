# Virelion CardiLearn

**A research-grade cardiac state representation model with explicit biological validation boundaries.**

CardiLearn is the learning layer of the Virelion cardiac ML ecosystem. The repository contains a structured transcriptomic representation prototype plus reproducible data, validation, interpretation, perturbation, and benchmarking infrastructure.

> **Scientific status:** CardiLearn is an executable research prototype, not a validated cardiac foundation model, clinical system, or evidence of regenerative efficacy. Real-data scientific claims must be demonstrated on locked held-out benchmarks and, where appropriate, independent biological validation.

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

The prototype is intentionally small enough for CPU/12-GB-RAM development experiments:

- 2,000 input genes
- 64-dimensional gene tokens
- 16 learned molecular program queries
- 128-dimensional shared latent
- 32-dimensional private latent

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
- **GSE130699** — neonatal mouse cardiomyocyte single-nucleus data spanning P1/P8 MI and sham conditions.
- **GSE217494** — human CITE-seq/GEX data spanning healthy and cardiomyopathy/MI contexts.

These are **candidate studies, not a locked benchmark**. Subject/sample relationships, technical replicates, conditions, regions, and timepoints must be reconciled before a scientific data lock.

Audit/materialize metadata:

```bash
python scripts/audit_real_data_pilot.py
python scripts/materialize_real_data_pilot.py
```

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
├── prototype/            # CardiLearn representation-learning prototype
├── interpretability/     # Step 13
├── perturbation/         # Step 14
├── benchmark_protocol.py # Step 15 evaluation contract
├── benchmark_suite.py    # Step 15 baseline implementations
├── real_data.py          # Real-data pilot contracts/parser
├── data.py               # Dataset contracts
├── splitting.py          # Leakage-safe splitting
├── validation.py         # Dataset integrity checks
├── provenance.py         # Run/data provenance
└── ...

docs/
configs/
scripts/
tests/
```

## Scientific maturity gate

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

## License

MIT
