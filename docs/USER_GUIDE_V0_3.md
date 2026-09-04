# CardiLearn v0.3 — User Guide

## Scope

CardiLearn is a research software package for reproducible cardiac machine-learning experiments. The repository currently combines a mature classical-ML infrastructure layer with the newer CardiLearn representation-learning prototype.

**Scientific status:** this package is research infrastructure. It is not a clinical system, diagnostic device, validated foundation model, or evidence of regenerative efficacy.

## Installation

Core installation:

```bash
pip install -e .
```

Development environment:

```bash
pip install -e '.[dev]'
```

Optional transcriptomic tooling:

```bash
pip install -e '.[bio]'
```

Optional PyTorch support for the representation prototype:

```bash
pip install -e '.[torch]'
```

## Command-line entry point

The package exposes the `cardilearn` command.

Inspect available baseline models:

```bash
cardilearn models --task classification
```

Audit a dataset before training:

```bash
cardilearn validate --data data.csv --target label
```

Train a baseline and save a run artifact:

```bash
cardilearn train \
  --data data.csv \
  --target label \
  --model logistic_regression \
  --output runs/example
```

Inspect a benchmark definition:

```bash
cardilearn benchmark-info --definition configs/benchmark_v1.yaml
```

Run a declared benchmark:

```bash
cardilearn benchmark \
  --data data.csv \
  --definition configs/benchmark_v1.yaml \
  --output runs/benchmark.json
```

## Python API

The top-level API intentionally exposes stable contracts rather than every internal implementation detail:

```python
import cardilearn

report = cardilearn.validate_dataset(dataset, target="label")
```

The representation prototype is lazy-loaded so users who only need metadata, validation, splitting, or benchmark infrastructure do not need PyTorch installed:

```python
from cardilearn.prototype import CardiLearnProto
```

Interpretability:

```python
from cardilearn.interpretability import attribute_latent, rank_genes
```

Perturbation prediction:

```python
from cardilearn.perturbation import PerturbationPredictor
```

Benchmarking:

```python
from cardilearn import BenchmarkSpec, rank_models
```

## Research workflow

The recommended workflow is:

```text
source data
   ↓
metadata audit
   ↓
DatasetSpec / FeatureManifest
   ↓
leakage-safe split
   ↓
train-only preprocessing
   ↓
baseline + CardiLearn training
   ↓
locked evaluation
   ↓
interpretability / perturbation analysis
   ↓
cross-species / external validation
   ↓
experiment artifact + provenance
```

Do not skip the metadata and split stages merely because a model can technically train without them.

## Real-data pilot

The current pilot is metadata-first. Use:

```bash
python scripts/audit_real_data_pilot.py
python scripts/materialize_real_data_pilot.py
```

These commands establish candidate dataset metadata and provenance. They do not make the candidate studies a final benchmark automatically.

Before a scientific claim, lock:

- source dataset versions;
- biological subject/sample relationships;
- feature universe;
- inclusion/exclusion rules;
- train/validation/test manifests;
- preprocessing fitted on training data only;
- random seeds;
- model configuration;
- evaluation metrics.

## Leakage policy

The fundamental unit of leakage protection is the biological unit, not the number of rows.

For single-cell and single-nucleus data, cells from the same biological sample must not be treated as independent biological replicates when assigning train/test membership. Whole studies and biological subjects should remain intact where the declared benchmark requires it.

If a preprocessing step can learn from the data, fit it only on the training partition.

## Interpretability policy

Interpretability outputs are hypothesis-generation tools.

Use multiple evidence types where possible:

1. gradient attribution;
2. masking/permutation sensitivity;
3. latent-factor analysis;
4. pathway/gene-set enrichment;
5. cross-species ortholog conservation;
6. counterfactual sensitivity.

Do not infer causality from attribution alone.

## Perturbation policy

The perturbation module predicts latent responses. A predicted response is not a causal effect estimate unless supported by an appropriate experimental design and independent validation.

Keep these concepts separate:

```text
prediction ≠ causation
attribution ≠ mechanism
latent similarity ≠ biological equivalence
```

## Benchmark policy

The benchmark suite compares declared baselines under a common protocol. Model selection must occur on validation data; the final test set remains locked until the experiment is frozen.

Do not report a benchmark result unless the corresponding dataset fingerprint, split manifest, configuration, seed set, and software version can be recovered.

## Reproducibility checklist

Before accepting an experiment:

- [ ] dataset provenance recorded
- [ ] dataset fingerprint recorded
- [ ] feature manifest recorded
- [ ] split manifest frozen
- [ ] no subject/sample overlap across partitions
- [ ] preprocessing fitted only on training data
- [ ] configuration saved
- [ ] random seeds saved
- [ ] package/environment version recorded
- [ ] model artifact saved
- [ ] metrics saved
- [ ] held-out test used only after model selection
- [ ] no unsupported biological/causal claims

## Limitations

The current repository is not yet a fully trained cross-species cardiac foundation model. The representation prototype remains small and computationally constrained, and real-data performance depends on future dataset curation and locked evaluation. The candidate pilot studies are not themselves sufficient to establish generalization across cardiac development, maturation, injury, and regeneration.

The correct progression is:

```text
software framework
   → data lock
   → training
   → benchmark
   → independent validation
   → biological interpretation
   → experimental validation
```
