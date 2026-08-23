# CardiBench → CardiLearn workflow

CardiLearn consumes benchmark definitions from the Virelion-CardiBench repository but does **not** copy or redistribute source datasets.

## 1. Register a benchmark definition

A definition must specify:

- `benchmark_id` and `version`
- task type
- target column
- biological grouping key
- split policy
- source dataset accessions
- primary metrics

The contract follows the CardiBench `BenchmarkDefinition` schema.

## 2. Materialize the dataset table

The table supplied to CardiLearn should contain at minimum:

```text
sample_id
biological_replicate_id
<target>
<model features...>
```

For multimodal datasets, align modalities on `sample_id` before training. Keep study, species, age, timepoint, anatomical zone, genotype, sex, cell context, and technical replicate identifiers as metadata columns when relevant.

## 3. Enforce biological split boundaries

For the ordinary within-study setting, the benchmark runner supports subject/donor/animal/technical-group policies.

For external generalization protocols—study-held-out, species-held-out, timepoint-held-out—the split must be materialized explicitly by CardiBench. CardiLearn deliberately refuses to infer such a split from arbitrary metadata because that can silently produce invalid scientific comparisons.

### Cardiac cell-level classification rule

Cells are observations, but cells from the same biological sample are not independent biological replicates. For classification benchmarks such as MI-vs-Sham, the biological sample is the grouping unit. CardiLearn therefore uses `StratifiedGroupKFold` through `cardilearn.splitting.make_classification_splitter(...)` whenever classification observations have biological groups.

This is intentionally different from ordinary `GroupKFold`: the splitter keeps each biological sample intact while attempting to preserve the class distribution across folds. A feasibility check requires enough biological groups in **each class** for the requested number of folds. If that is impossible, CardiLearn raises an error rather than silently generating one-class validation folds where AUROC is undefined.

Regression retains the existing generic grouped behavior.

## 4. Train on development data

```bash
python -m cardilearn benchmark \
  --data prepared_features.csv \
  --definition configs/benchmarks/mi-vs-sham-cardiobench.json \
  --model logistic_regression \
  --output runs/mi-vs-sham.json
```

Model selection remains development-only. The final test partition is not used for tuning.

For the MI-vs-Sham benchmark, the primary development dataset is GSE153480 and GSE216211 is an external validation dataset. MI-E is excluded from the primary GSE216211 binary task and must not influence feature selection or model selection.

## 5. Export frozen predictions

Use `cardilearn.io.export_predictions()` for the legacy-compatible interface or `cardilearn.io.export_frozen_predictions()` for the full `cardilearn.predictions.v1` interchange schema. Frozen predictions should retain sample ID, observation ID, true label, predicted label, probability, fold, dataset, and model.

## 6. Independent evaluation

CardiEval should ingest the frozen predictions and calculate comparative statistics, confidence intervals, calibration analyses, subgroup analyses, and model-vs-model significance tests. CardiLearn must not select a model on the basis of CardiEval test results.

Confidence intervals for biological-replicate evaluation must resample biological samples (or studies when multiple studies are available), not individual cells.

## Two-track MI-vs-Sham protocol

### Track 1 — exploratory cell-level representation

```text
GSE153480
  ↓
QC
  ↓
fold-specific feature selection
  ↓
representation
  ↓
StratifiedGroupKFold
  ↓
model
  ↓
validation predictions
```

This asks whether molecular representations distinguish MI from Sham at the cell-observation level. It is explicitly exploratory because cell count is not biological replicate count.

### Track 2 — biological-replicate benchmark

The primary scientifically defensible track aggregates gene counts within each biological sample (pseudobulk/sample-level representation), fits normalization/feature selection only on training samples, and evaluates with the biological sample as the independent statistical unit.

## Priority cardiac tasks

### MI vs sham/reference

This is the first canonical benchmark family. The CardiBench registry currently identifies candidate public injury datasets including GSE153480, GSE135310, GSE216211, GSE106472, and GSE269054. Their source metadata must be re-verified at materialization time, especially biological replicate and pooling structure.

Recommended evaluations:

1. within-study subject-level MI vs sham/reference
2. study-held-out external generalization
3. timepoint-held-out temporal generalization
4. cell-context-aware evaluation to detect composition shortcuts
5. species-held-out transfer where biologically appropriate

Do not merge permanent MI with I/R, do not merge genotype perturbations into the primary injury label, and do not allow cells/nuclei or technical replicates from the same biological subject to cross evaluation boundaries.

## Candidate future task families

- cardiac regeneration vs non-regeneration
- developmental/maturation-stage prediction
- post-infarction temporal-state prediction
- injury-zone/border-zone classification
- cross-species cardiac-state transfer
- multimodal phenotype prediction
