# Virelion CardiLearn

**A reproducible research prototype for large-scale cardiac molecular-state representation learning.**

CardiLearn is the learning layer of the Virelion cardiac ML ecosystem. It is designed to learn transcriptome-scale representations of cardiac cell states across development, maturation, injury, regeneration, and species, with explicit leakage control and held-out biological evaluation.

> **Scientific status:** CardiLearn is an executable research model under development, not a validated cardiac foundation model, clinical system, or evidence of regenerative efficacy. The repository now contains both a small integration model and a large research architecture. The large architecture targets ~20,000 genes, 1,024-dimensional gene/program representations, 32 learned molecular programs, six deep Transformer blocks, and 512-dimensional shared state. Its scale makes it biologically substantive enough to test meaningful transcriptome-level hypotheses, but scale alone is not biological validation. Claims require locked real-data training, held-out benchmarks, cross-study/species validation, and independent biological validation where appropriate.

## Research model: from prototype to serious scale

The small `CardiLearnProto` model remains useful for software development. The research path is now `CardiLearnLarge`.

### CardiLearnLarge reference scale

- **~20,000 transcriptome genes** rather than a 2,000-gene toy subset;
- **1,024-dimensional gene/program representations**;
- **32 learned molecular programs**;
- **6 Transformer blocks** over program tokens;
- **16 attention heads**;
- **512-dimensional shared biological state**;
- **128-dimensional private state**;
- transcriptome-scale reconstruction;
- species and assay context conditioning;
- gene→program cross-attention followed by deep program-level interaction modeling.

This is a deliberate architectural shift: the model spends capacity on **gene-network context and molecular programs**, rather than simply inflating a conventional MLP. Dense self-attention over all ~20k genes would be computationally prohibitive, so CardiLearn first compresses the transcriptome through learned program queries and applies depth to the resulting molecular-program sequence.

The large architecture is intended to be trained on a broad, carefully curated corpus rather than on one small cardiac dataset. Modern single-cell foundation models demonstrate that meaningful transcriptomic representation learning operates at tens to hundreds of millions of cells and roughly 20k-gene vocabulary scales; CardiLearn's research target is therefore deliberately aligned with that regime while remaining cardiac-specialized. citeturn0search1turn0search3turn0search10

## Scope: scale first, claims later

The project separates **model scale**, **software maturity**, and **scientific validity**. A large model can learn more structure without proving that the learned structure is biologically correct. CardiLearn therefore uses:

```text
large-scale architecture
   ↓
large, diverse, provenance-locked corpus
   ↓
train-only preprocessing
   ↓
biological-family-safe split
   ↓
pretraining
   ↓
locked downstream benchmarks
   ↓
unseen-study validation
   ↓
unseen-species validation
   ↓
mechanistic / experimental validation
```

This is intentionally stricter than simply reporting training loss. Recent evaluations of single-cell foundation models show that model scale does not guarantee reliable zero-shot biological performance, reinforcing the need for strong external benchmarks and simple baselines. citeturn0search7

## Research model

```text
transcriptome (~20k genes)
        ↓
gene/value embeddings
        ↓
32 learned molecular program queries
        ↓
program-level Transformer stack
        ↓
context-conditioned molecular state
        ↓
shared + private latent representation
        ├── maturation
        ├── injury / disease state
        ├── cell identity
        ├── reconstruction
        └── downstream perturbation / transition models
```

Implementation: `cardilearn/prototype/model.py`.

## Research capabilities

| Layer | Purpose |
|---|---|
| Data + validation | explicit contracts, provenance, integrity checks, leakage-safe biological splits |
| Large representation | transcriptome-scale shared/private cardiac molecular-state representation |
| Molecular programs | learned latent programs intended to capture recurring gene-network structure |
| Interpretability | attribution, masking/permutation, enrichment, counterfactuals, ortholog conservation |
| Perturbation | predictive latent-response modeling with uncertainty |
| Benchmarking | matched baseline comparisons with locked evaluation rules |

Detailed architecture: `docs/ARCHITECTURE.md`.

Research usage guide: `docs/USER_GUIDE_V0_3.md`.

Documentation standard: `docs/STEP17_DOCUMENTATION.md`.

## Scientific formulation

The learning objectives include molecular reconstruction, technical-view invariance, anti-collapse regularization, biological-state supervision, maturation ordering/regression, and injury-state prediction. The next research expansion should add explicit masked-gene modeling and cross-study/cross-species objectives so that the large model learns transferable molecular structure rather than merely reconstructing its training cohorts.

Regeneration is not implemented as a hand-written marker score. A regenerative representation must be demonstrated by generalization across independent injury/regeneration studies and, ultimately, experimental evidence.

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
4. CardiLearnLarge

The large model only earns its complexity if it improves held-out biological generalization and survives shortcut/robustness tests. The small `CardiLearnProto` is retained as an engineering smoke-test model, not as the scientific endpoint.

## Training scale and hardware

The large architecture is no longer designed around 12-GB CPU development. A serious run should be treated as GPU research training.

Recommended starting target:

- NVIDIA GPU: **24 GB VRAM or more**;
- system RAM: **32–64 GB**;
- local NVMe: **500 GB–1 TB+** for processed matrices, caches, checkpoints, and manifests;
- CPU: **8–16+ cores**;
- mixed precision and gradient accumulation for memory control.

A 13-GB Colab RAM environment can still be useful for preprocessing, metadata audits, smoke tests, and the small model. It should **not** be treated as the target environment for full-scale CardiLearnLarge pretraining.

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

For real-data research training, `scripts/train_real_data.py` now defaults to `CardiLearnLarge`; `--model-size proto` is available for lightweight software validation.

## Repository structure

```text
cardilearn/
├── prototype/            # CardiLearnProto + CardiLearnLarge
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
architecture
   ↓
corpus audit
   ↓
data lock
   ↓
large-scale pretraining
   ↓
locked benchmark
   ↓
unseen-study validation
   ↓
unseen-species validation
   ↓
mechanistic interpretation
   ↓
independent biological validation
```

A passing CI build demonstrates software correctness, not biological validity. A large parameter count demonstrates capacity, not biological significance.

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

The architecture is now large enough that **the bottleneck is no longer model size**. The next major milestone is the training corpus: a genuinely broad, independently sourced cardiac transcriptomic corpus with reviewed biological hierarchy, explicit orthology, controlled modality boundaries, and frozen study-family splits. Only after that corpus is locked should we spend substantial GPU time pretraining CardiLearnLarge.

## License

MIT
