# Virelion CardiLearn

**Real-data cardiac machine learning for reproducible learning, benchmarking, and transfer.**

CardiLearn is the model-learning layer of the Virelion cardiac ML ecosystem. It is designed to train models on curated cardiac datasets, preserve leakage-safe evaluation boundaries, produce reproducible artifacts, and expose models through stable interfaces that downstream components can consume.

## What CardiLearn is for

CardiLearn focuses on learning from real cardiac observations rather than generating challenge scenarios. The initial architecture supports:

- tabular, molecular, imaging-derived, ECG-derived, and phenotype feature tables
- classification and regression tasks
- patient/sample/group-aware train/validation/test splitting
- preprocessing fitted only on training data
- deterministic baselines for fast validation
- artifact serialization with schema and provenance metadata
- metrics suitable for downstream CardiEval evaluation

## Design principles

1. **No leakage by construction.** Groups such as patient, donor, animal, study, or experiment can define split boundaries.
2. **Dataset contracts before models.** Every training run starts from explicit feature, target, group, and metadata definitions.
3. **Baseline first.** A strong deterministic baseline is established before more complex models are introduced.
4. **Reproducibility is part of the model.** Configuration, versions, split assignments, metrics, and model artifacts are saved together.
5. **Evaluation stays independent.** CardiLearn trains models; CardiEval remains the independent evaluation layer.

## Repository layout

```text
cardilearn/
  config.py          # typed training configuration
  data.py            # dataset contracts and loading utilities
  splitting.py       # leakage-safe deterministic splitting
  preprocessing.py   # train-fitted preprocessing
  models.py          # baseline model registry
  metrics.py         # task metrics
  training.py        # end-to-end training orchestration
  artifacts.py       # portable run/model metadata
  cli.py             # command-line entry point
  __main__.py        # python -m cardilearn

tests/               # unit tests for core invariants
configs/              # example experiment configurations
.github/workflows/    # continuous integration
```

## Quick start

```bash
python -m pip install -e .
python -m cardilearn --help
pytest -q
```

A minimal configuration is included at `configs/example-classification.json`.

## Current milestone

The first CardiLearn milestone establishes a trustworthy learning core: explicit dataset schemas, group-aware splitting, train-only preprocessing, baseline models, metrics, run manifests, and CI. Neural and modality-specific trainers will build on these contracts rather than bypassing them.

## Ecosystem

- **CardiAgent** — cardiac challenge-agent generation
- **CardiVex** — challenge detection and characterization
- **CardiBench** — curated benchmarks and canonical splits
- **CardiLearn** — real-data model training
- **CardiEval** — independent evaluation and statistical comparison
- **CardiAtlas** — literature and cardiac omics/phenotype knowledge base
- **CardiSim** — synthetic cardiac trajectory simulation
- **CardiTrace** — provenance and reproducibility
- **CardiBridge** — cross-component schemas and APIs

## License

MIT
