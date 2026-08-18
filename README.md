# Virelion CardiLearn

**Real-data cardiac machine learning for reproducible learning, benchmarking, and transfer.**

CardiLearn is the learning layer of the Virelion cardiac ML ecosystem. It trains models from real cardiac observations while enforcing scientific boundaries between training, model selection, and held-out evaluation.

## Current capabilities

### Data and provenance
- Explicit dataset contracts with target, sample, and subject/study grouping metadata.
- CSV and modality-table adapters for cardiac feature matrices.
- Dataset fingerprints and reproducibility metadata.
- Multimodal alignment by unique sample ID.
- Integrity checks for missingness, duplicate IDs, missing targets, and group leakage.

### Learning
- Classification: logistic regression, histogram gradient boosting, MLP.
- Regression: ridge, histogram gradient boosting, MLP.
- Train-fitted preprocessing for mixed numeric/categorical tables.
- Group-aware train/validation/test splitting.
- Development-only cross-validation and model selection.
- High-dimensional omics feature selection + PCA.
- ECG/time-series summary feature extraction.
- Probability calibration and calibration metrics.
- Validation-set permutation importance.
- PCA representation-learning baseline.

### Interoperability
- Portable model artifacts and manifests.
- Stable `cardilearn.predictions.v1` prediction export for CardiEval.
- Configuration and split assignments saved with runs.
- Test data are **not evaluated during ordinary training**. Held-out evaluation is explicit and verifies the dataset fingerprint.

## Scientific contract

1. **Groups do not cross evaluation boundaries.** Patient, donor, animal, experiment, or study identifiers can define partitions.
2. **The test partition is sacred.** Model selection occurs using training/development data only.
3. **Preprocessing is fitted on training data.** This includes imputation, scaling, supervised feature selection, and dimensionality reduction.
4. **Every result is reproducible.** Configuration, feature schema, split indices, dataset fingerprint, metrics, and model artifact travel together.
5. **Evaluation is independent.** CardiLearn produces models and frozen predictions; CardiEval owns independent statistical comparison.

## Repository layout

```text
cardilearn/
  adapters.py       # modality-table loading and multimodal alignment
  artifacts.py      # model/run serialization
  benchmarks.py     # leakage-safe cross-validation
  calibration.py    # calibration and ECE/Brier metrics
  config.py         # experiment configuration
  data.py           # dataset contracts
  ecg.py            # ECG/time-series summary features
  explainability.py # validation-set permutation importance
  io.py             # CardiEval prediction interchange
  metrics.py        # classification/regression metrics
  models.py         # model registry
  neural.py         # MLP neural baseline
  omics.py          # high-dimensional omics preprocessing
  provenance.py     # fingerprints/runtime metadata
  selection.py      # development-only model selection
  splitting.py      # leakage-safe deterministic splitting
  training.py       # end-to-end training and held-out evaluation
  validation.py     # dataset integrity checks
configs/            # example experiments
configs/benchmarks/ # benchmark templates

docs/               # architecture and experiment protocol
tests/              # scientific invariants and integration tests
.github/workflows/  # CI
```

## Quick start

```bash
python -m pip install -e .
cardilearn models --task classification
cardilearn train --data data.csv --target outcome --group patient_id --model hist_gradient_boosting --output runs/example
pytest -q
```

To evaluate a frozen held-out result, use `evaluate_held_out_test` after model selection. Do not incorporate test metrics into tuning or feature selection.

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
