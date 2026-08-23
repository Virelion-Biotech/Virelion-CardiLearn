# Virelion CardiLearn

**Real-data cardiac machine learning for reproducible learning, benchmarking, and transfer.**

CardiLearn is the learning layer of the Virelion cardiac ML ecosystem. It trains models from real cardiac observations while enforcing scientific boundaries between training, model selection, and held-out evaluation.

## Current capabilities

### Data and provenance
- Explicit dataset contracts with target, sample, and subject/study grouping metadata.
- CSV and modality-table adapters for cardiac feature matrices.
- Reproducible NCBI GEO acquisition with SHA-256 manifests; source data are never committed to Git.
- Dataset fingerprints and reproducibility metadata.
- Multimodal alignment by unique sample ID.
- Integrity checks for missingness, duplicate IDs, missing targets, and group leakage.

### Learning
- Classification: logistic regression, histogram gradient boosting, MLP.
- Regression: ridge, histogram gradient boosting, MLP.
- Train-fitted preprocessing for mixed numeric/categorical tables.
- Group-aware train/validation/test splitting.
- Classification CV uses `StratifiedGroupKFold` when biological groups are supplied.
- Development-only cross-validation and model selection.
- High-dimensional omics feature selection + PCA.
- ECG/time-series summary feature extraction.
- Probability calibration and calibration metrics.
- Validation-set permutation importance.
- PCA representation-learning baseline.

### CardiBench benchmark integration
- CardiBench-compatible benchmark-definition loader.
- Explicit biological split policies and external-level holdouts.
- MI/sham priority source manifests.
- Reproducible model-matrix execution over prepared cardiac feature tables.
- Benchmark versions, dataset fingerprints, and model configurations are recorded with results.

The first concrete benchmark matrix targets myocardial infarction versus sham/reference using public scRNA-seq datasets including GSE153480 and GSE216211. P1/P8, timepoint, MI-E, study, and biological-group metadata remain explicit rather than being collapsed into a single label.

## MI-vs-Sham scientific protocol

The primary benchmark definition is `configs/benchmarks/mi-vs-sham-cardiobench.json`. GSE153480 is the development dataset and GSE216211 is external validation. MI-E is excluded from the primary GSE216211 binary task.

For cell-level exploratory representation, cells are observations but the biological sample is the independent grouping unit. CardiLearn uses `cardilearn.splitting.make_classification_splitter(...)`, backed by `StratifiedGroupKFold`, so a biological sample can never cross folds while MI/Sham balance is optimized. The splitter refuses infeasible configurations rather than silently creating one-class validation folds.

The primary scientifically defensible track is sample-level/pseudobulk evaluation. Cells from the same biological sample must not be counted as independent biological replicates, and confidence intervals must resample biological samples rather than individual cells.

See `docs/CARDIOBENCH_WORKFLOW.md` and `docs/MI_SHAM_BENCHMARK.md` for the complete protocol.

## Scientific contract

1. **Groups do not cross evaluation boundaries.** Patient, donor, animal, experiment, or study identifiers can define partitions.
2. **The test partition is sacred.** Model selection occurs using training/development data only.
3. **Preprocessing is fitted on training data.** This includes imputation, scaling, supervised feature selection, and dimensionality reduction.
4. **Every result is reproducible.** Configuration, feature schema, split indices, dataset fingerprint, metrics, and model artifact travel together.
5. **Evaluation is independent.** CardiLearn produces models and frozen predictions; CardiEval owns independent statistical comparison.
6. **Source ambiguity stops the pipeline.** Metadata are validated instead of guessed.
7. **Cells are not biological replicates.** Cell-level observations remain grouped by biological sample for cardiac injury benchmarks.

## Materializing the first real benchmark

Raw and processed source datasets are intentionally not stored in this repository. Materialize them locally:

```bash
python -m pip install -e '.[bench,bio]'
python scripts/materialize_geo.py --accession GSE153480 --kind family_soft --cache data/raw
python scripts/materialize_geo.py --accession GSE153480 --kind raw_tar --cache data/raw
python scripts/materialize_geo.py --accession GSE216211 --kind family_soft --cache data/raw
python scripts/materialize_geo.py --accession GSE216211 --kind raw_tar --cache data/raw
```

After source metadata have been resolved and expression data have been reduced to one row per biological sample/group, place the feature tables at:

```text
data/prepared/GSE153480/sample_features.csv
data/prepared/GSE216211/sample_features.csv
```

Each table must contain at minimum:

```text
sample_id
biological_group_id
study_id
injury_label
x1 ... xN
```

Then run the real model matrix:

```bash
python scripts/run_mi_sham_matrix.py
```

Results are written to `runs/mi-sham-matrix/results.json`. Final held-out test evaluation remains explicit and must not be used for tuning.

## Repository layout

```text
cardilearn/
  adapters.py        # modality-table loading and multimodal alignment
  artifacts.py       # model/run serialization
  benchmark_matrix.py# reproducible candidate-model matrix
  benchmarks.py      # leakage-safe cross-validation
  cardiobench.py     # CardiBench-compatible definitions and holdouts
  calibration.py     # calibration and ECE/Brier metrics
  config.py          # experiment configuration
  data.py            # dataset contracts
  ecg.py             # ECG/time-series summary features
  explainability.py  # validation-set permutation importance
  geo.py             # reproducible NCBI GEO acquisition
  io.py              # CardiEval prediction interchange
  metrics.py         # classification/regression metrics
  models.py          # model registry
  neural.py          # MLP neural baseline
  omics.py           # high-dimensional omics preprocessing
  provenance.py      # fingerprints/runtime metadata
  selection.py       # development-only model selection
  splitting.py       # leakage-safe deterministic splitting
  training.py        # training and held-out evaluation
  validation.py      # dataset integrity checks
configs/benchmarks/  # benchmark and source manifests
scripts/              # GEO materialization and benchmark execution
docs/                 # architecture and experiment protocol
tests/                # scientific invariants and integration tests
.github/workflows/    # CI
```

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
