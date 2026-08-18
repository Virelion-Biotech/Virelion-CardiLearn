# Virelion CardiLearn

**Real-data cardiac machine learning for reproducible learning, benchmarking, and transfer.**

CardiLearn is the learning layer of the Virelion cardiac ML stack. It turns heterogeneous cardiac observations into reproducible models while enforcing scientific evaluation boundaries: subject-aware splitting, train-only preprocessing, explicit dataset contracts, provenance, and independent evaluation handoff to CardiEval.

## What it can do now

- **Tabular:** phenotype, clinical, functional, ECG-derived and imaging-derived feature tables.
- **Omics:** sample × feature matrices with stable transformation hooks.
- **ECG / time series:** waveform containers and summary feature extraction.
- **Imaging:** array-based image contracts and normalization hooks.
- **Tasks:** binary/multiclass classification and regression baselines.
- **Splitting:** group-aware random splits and forward-chaining longitudinal validation.
- **Validation:** duplicate-ID, missingness, target integrity, and group-leakage checks.
- **Modeling:** logistic regression, ridge regression, and histogram gradient boosting.
- **Evaluation:** cross-validation, calibration, permutation importance, task metrics, and benchmark tables.
- **Representation learning:** dependency-light PCA embeddings with a stable interface for future deep encoders.
- **Reproducibility:** dataset cards, run manifests, configuration, and registry layout.

## Scientific guardrails

CardiLearn deliberately separates model development from final assessment. The validation set is for model selection; the held-out test set is not touched until the model is frozen. Subject-level groups should normally define boundaries, and longitudinal studies can use chronological forward validation. CardiEval can then independently score exported predictions.

## Repository layout

```text
cardilearn/
  config.py          # experiment configuration
  schema.py          # dataset + feature contracts
  data.py            # dataframe contracts
  loaders.py         # CSV/TSV/Parquet/Feather ingestion
  modalities.py      # omics, waveform, image containers
  featurization.py   # modality-level summaries
  splitting.py       # group-aware splitting
  temporal.py        # forward-chaining validation
  preprocessing.py   # train-fitted preprocessing
  models.py          # baseline model registry
  metrics.py         # task metrics
  advanced.py        # CV, calibration, permutation importance
  embeddings.py      # representation-learning baseline
  validation.py      # dataset integrity checks
  benchmark.py       # model comparison protocol
  dataset_card.py    # provenance metadata
  artifacts.py       # portable artifacts
  registry.py        # run/model registry
  training.py        # training orchestration
  cli.py             # command-line interface

tests/               # unit + scientific invariant tests
configs/             # reproducible experiment examples
.github/workflows/   # CI
```

## Quick start

```bash
python -m pip install -e .
python -m cardilearn --help
pytest -q
```

Example configuration: `configs/example-classification.json`.

## Roadmap

The foundation is intentionally modality-neutral. The next layers are dataset-specific loaders for CardiBench-compatible cardiac datasets, high-dimensional omics reduction, ECG foundation encoders, image encoders, multimodal fusion, hyperparameter optimization, uncertainty estimation, model cards, and direct CardiEval export.

## Ecosystem

- **CardiAgent** — challenge-agent generation
- **CardiVex** — challenge detection and characterization
- **CardiBench** — curated datasets and canonical splits
- **CardiLearn** — real-data model training
- **CardiEval** — independent evaluation and statistical comparison
- **CardiAtlas** — cardiac literature and knowledge base
- **CardiSim** — synthetic trajectory simulation
- **CardiTrace** — provenance and reproducibility
- **CardiBridge** — cross-component schemas and APIs

## License

MIT
