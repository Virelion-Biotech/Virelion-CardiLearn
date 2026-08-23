# CardiBench workflow in CardiLearn

CardiLearn is the training and prediction layer; CardiEval remains the independent evaluation layer.

```text
raw data
  ↓
QC
  ↓
sample metadata
  ↓
biological grouping
  ↓
fold-specific preprocessing
  ↓
StratifiedGroupKFold (classification)
  ↓
model
  ↓
frozen predictions
  ↓
CardiEval
```

## Biological grouping is mandatory for cardiac cell benchmarks

Cells are observations, but cells from the same biological sample are not independent biological replicates. A sample must remain intact during validation. For MI-vs-Sham classification, CardiLearn therefore uses `StratifiedGroupKFold` rather than ordinary `GroupKFold`: the group is the biological sample and the target is MI/Sham.

The splitter is exposed as `cardilearn.splitting.make_classification_splitter(...)`. It performs a feasibility check before yielding folds. If either class has fewer biological groups than `n_splits`, the benchmark stops with a clear error rather than producing validation folds in which AUROC is undefined.

## Two benchmark tracks

### Track 1 — exploratory cell-level representation

This track asks whether molecular representations distinguish MI from Sham at the cell-observation level. Feature selection, scaling, and dimensionality reduction must be fitted inside each training fold. Cells from a biological sample never cross a fold.

This result is explicitly exploratory because thousands of cells do not create thousands of independent biological replicates.

### Track 2 — biological-replicate benchmark

This is the primary scientifically defensible track. Counts are aggregated within each biological sample (pseudobulk/sample-level representation), normalization and feature selection are learned from training samples only, and evaluation is performed with the biological sample as the statistical unit.

Confidence intervals must therefore resample biological samples, not individual cells.

## External validation

GSE153480 is the development dataset and GSE216211 is external validation. MI-E is excluded from the primary GSE216211 MI-vs-Sham task. No external sample may influence feature selection, hyperparameter tuning, model selection, or representation fitting.

## Prediction contract

Completed runs should export frozen predictions using `cardilearn.predictions.v1`, retaining dataset, model, fold/split, biological sample, observation identifier, true label, predicted label, and probability where applicable. CardiEval consumes these frozen predictions for independent metrics, confidence intervals, calibration, and statistical comparisons.
