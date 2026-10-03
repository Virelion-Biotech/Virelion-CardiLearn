# MI-vs-Sham benchmark protocol

## Scientific question

Can molecular representations distinguish myocardial infarction (MI) from Sham/reference in cardiac single-cell data without treating cells from one biological sample as independent replicates?

## Dataset roles

- **GSE153480:** primary development/training dataset.
- **GSE216211:** external validation dataset.
- **MI-E:** excluded from the primary GSE216211 binary MI-vs-Sham task.

Dataset manifests live under `data/manifests/`. Raw GEO archives are not committed.

## Track 1 — exploratory cell-level representation

```text
GSE153480
  ↓
QC
  ↓
feature selection fitted only on training fold
  ↓
representation
  ↓
StratifiedGroupKFold
  ↓
model
  ↓
validation predictions
```

The grouping unit is the biological sample. `StratifiedGroupKFold` simultaneously keeps each sample intact and attempts to balance MI/Sham across validation folds. A feasibility check rejects configurations that cannot supply enough biological groups per class.

The pilot numbers previously observed with ordinary grouped CV are sanity-check references only; they are not benchmark results and must never be copied into result artifacts.

## Track 2 — biological-replicate benchmark

This is the primary benchmark. For each biological sample, aggregate gene counts, normalize appropriately, transform, fit feature selection on training samples only, and train the model. Evaluate at sample level. Do not report cell counts as independent biological n.

## Model matrix

Candidate models must receive exactly the same splits:

- Logistic Regression
- Ridge where appropriate
- Histogram Gradient Boosting
- MLP

Primary metrics: AUROC, AUPRC, balanced accuracy.

Secondary metrics: accuracy, F1, sensitivity, specificity, Brier score, and calibration/ECE.

## Leakage controls

Feature selection, PCA/SVD, scaling, and other learned transformations must be fitted only on training data. External validation data are never used for tuning or model selection. Cell-level random splitting is prohibited.

## Statistical uncertainty

For biological-replicate evaluation, confidence intervals should resample biological samples (or studies when multiple studies are available). Bootstrapping individual cells is not a valid substitute for biological-replicate uncertainty.

## CardiEval boundary

CardiLearn trains and freezes predictions. CardiEval independently calculates final statistical metrics, confidence intervals, calibration, and model comparisons from `cardilearn.predictions.v1`.
