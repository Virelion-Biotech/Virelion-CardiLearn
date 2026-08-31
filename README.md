# Virelion CardiLearn

**Reproducible machine learning for real cardiac datasets, with explicit biological validation boundaries.**

CardiLearn is the learning layer of the Virelion cardiac ML ecosystem. It provides leakage-aware data contracts, preprocessing, model training, benchmark execution, transfer evaluation, calibration, explainability, and reproducible artifacts for cardiac ML research.

> **Current status:** software and benchmark infrastructure are substantially implemented. Real-data execution remains intentionally data-dependent; this repository does not commit source datasets or claim benchmark performance that has not been independently materialized and verified.

## Scope

CardiLearn is designed for research workflows in which the biological unit of inference matters. It separates observations from independent biological groups, keeps model development separate from held-out evaluation, and records provenance alongside outputs.

It is **not** a clinical decision-support system and does not establish clinical validity, diagnostic performance, or prospective utility by itself.

## Implemented capabilities

### Data, schemas, and provenance

- Explicit dataset contracts with target, sample, subject, donor, animal, study, and grouping metadata.
- CSV and modality-table adapters for cardiac feature matrices.
- Multimodal alignment by unique sample identifiers.
- Dataset integrity checks for missingness, duplicate identifiers, missing targets, and group leakage.
- Reproducibility metadata and deterministic dataset fingerprints.
- Reproducible NCBI GEO acquisition with SHA-256 manifests; source data remain outside Git.
- Dataset/model cards and serialized run artifacts.

### Learning

- Classification: logistic regression, histogram gradient boosting, and MLP.
- Regression: ridge, histogram gradient boosting, and MLP.
- Train-fitted preprocessing for mixed numeric/categorical tables.
- Deterministic group-aware train/validation/test splitting.
- `StratifiedGroupKFold` support for biological classification when feasible.
- Development-only cross-validation and model selection.
- High-dimensional omics preprocessing with train-only feature selection and PCA.
- ECG/time-series summary feature extraction.
- Probability calibration with ECE/Brier-style metrics.
- Validation-set permutation importance.
- PCA representation-learning baseline.

### CardiBench integration

- CardiBench-compatible benchmark-definition loading and validation.
- Explicit biological split policies and external-level holdout contracts.
- MI/sham priority source manifests.
- Reproducible candidate-model matrix execution over prepared feature tables.
- Benchmark versions, dataset fingerprints, model configurations, and predictions can travel with results.
- Cross-study transfer support with training-only feature selection.
- Bootstrap confidence intervals and permutation-null analysis for transfer uncertainty.

## Primary benchmark: MI vs sham/reference

The first concrete benchmark track evaluates myocardial infarction versus sham/reference using public cardiac transcriptomic datasets. The current protocol defines **GSE153480 as the development cohort** and **GSE216211 as an external validation cohort**; MI-E is excluded from the primary GSE216211 binary task. Additional cohorts, including GSE153485, are being treated as validation evidence rather than silently pooled into development.

Cell-level data may be used for exploratory representation learning, but the primary inferential unit is the biological sample. Cells from the same biological sample must not cross evaluation boundaries or be counted as independent biological replicates.

The benchmark definition is:

```text
configs/benchmarks/mi-vs-sham-cardiobench.json
```

Protocol documentation:

- `docs/CARDIOBENCH_WORKFLOW.md`
- `docs/MI_SHAM_BENCHMARK.md`

## Scientific safeguards

CardiLearn enforces the following principles in code and benchmark contracts:

1. **Biological groups do not cross evaluation boundaries.** Patient, donor, animal, experiment, or other declared groups remain intact.
2. **The test partition is protected.** Model selection occurs on development data only.
3. **Preprocessing is fitted on training data.** Imputation, scaling, supervised feature selection, and dimensionality reduction cannot inspect held-out observations.
4. **Cell counts are not replicate counts.** Biological confidence intervals must be based on independent biological units.
5. **Source ambiguity is a failure condition.** Metadata are validated rather than guessed.
6. **External validation is explicit.** Study-level or other external holdouts are represented by benchmark contracts/manifests rather than inferred ad hoc.
7. **Results are provenance-bound.** Configuration, schema, split information, fingerprints, metrics, and artifacts are intended to remain traceable.
8. **CardiLearn and CardiEval have separate responsibilities.** CardiLearn produces models and frozen predictions; CardiEval performs independent statistical comparison.

## What is complete vs data-dependent

### Implemented without requiring source datasets

- Core package and CLI architecture.
- Dataset/schema and integrity contracts.
- Leakage-safe splitting.
- Train-fitted preprocessing and model registry.
- Classification/regression training orchestration.
- Benchmark definitions and candidate-model matrix.
- Omics and ECG feature layers.
- Calibration and explainability utilities.
- Cross-study transfer machinery.
- Provenance and artifact serialization.
- Unit/invariant tests and continuous integration.

### Requires real data before it can be claimed as validated

- Final benchmark performance tables.
- Fully materialized cross-cohort results.
- Dataset-specific feature-selection stability.
- External validation confidence intervals based on the actual cohorts.
- Biological interpretation of learned features.
- Claims of superiority over existing cardiac ML methods.

This distinction is deliberate: **absence of committed data is not evidence of absence of capability, but it also is not permission to report unverified performance.**

## Reproducing the first benchmark

Raw and processed source datasets are intentionally not stored in this repository. Materialize them locally:

```bash
python -m pip install -e '.[bench,bio]'
python scripts/materialize_geo.py --accession GSE153480 --kind family_soft --cache data/raw
python scripts/materialize_geo.py --accession GSE153480 --kind raw_tar --cache data/raw
python scripts/materialize_geo.py --accession GSE216211 --kind family_soft --cache data/raw
python scripts/materialize_geo.py --accession GSE216211 --kind raw_tar --cache data/raw
```

After source metadata are resolved and expression data are reduced to one row per biological sample/group, prepare:

```text
data/prepared/GSE153480/sample_features.csv
data/prepared/GSE216211/sample_features.csv
```

Minimum required columns are:

```text
sample_id
biological_group_id
study_id
injury_label
x1 ... xN
```

Then run:

```bash
python scripts/run_mi_sham_matrix.py
```

The expected result location is:

```text
runs/mi-sham-matrix/results.json
```

Final held-out evaluation must remain frozen and must not be used for tuning.

## Development and quality checks

Install development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Run the complete test suite:

```bash
pytest -q
```

Run linting:

```bash
ruff check cardilearn tests scripts
```

The GitHub Actions workflow tests Python 3.10–3.12 on pushes to `main` and pull requests. CI is intended to verify software invariants; it does not substitute for real-data scientific validation.

## Repository layout

```text
cardilearn/
  adapters.py          # modality-table loading and multimodal alignment
  artifacts.py         # model/run serialization
  benchmark_matrix.py  # reproducible candidate-model matrix
  benchmark_runner.py  # single benchmark execution
  benchmarks.py        # leakage-safe cross-validation
  cardiobench.py       # CardiBench definitions and holdouts
  calibration.py       # calibration and uncertainty metrics
  config.py            # experiment configuration
  data.py              # dataset contracts
  ecg.py               # ECG/time-series feature extraction
  explainability.py    # validation-set permutation importance
  geo.py               # reproducible NCBI GEO acquisition
  io.py                # CardiEval prediction interchange
  metrics.py           # classification/regression metrics
  models.py            # model registry
  neural.py            # MLP baseline
  omics.py             # high-dimensional omics preprocessing
  provenance.py        # fingerprints/runtime metadata
  selection.py         # development-only model selection
  splitting.py         # leakage-safe deterministic splitting
  training.py          # training and held-out evaluation
  validation.py        # dataset integrity checks
configs/benchmarks/    # benchmark and source manifests
docs/                  # architecture and experiment protocols
scripts/               # GEO materialization and benchmark execution
tests/                 # scientific invariants and integration tests
.github/workflows/     # continuous integration
```

## Ecosystem

- **CardiAgent** — cardiac challenge-agent generation
- **CardiVex** — challenge detection and characterization
- **CardiBench** — curated benchmarks and canonical splits
- **CardiLearn** — real-data model training
- **CardiEval** — independent evaluation and statistical comparison
- **CardiAtlas** — cardiac literature and omics/phenotype knowledge base
- **CardiSim** — synthetic cardiac trajectory simulation
- **CardiTrace** — provenance and reproducibility
- **CardiBridge** — cross-component schemas and APIs

## License

MIT
