# Virelion-CardiLearn Full Validation Report

Date: 2026-08-30
Operator: automated validation (Grok)

## Dataset materialization
- GSE153480_RAW.tar downloaded from NCBI GEO FTP
- Actual SHA-256: 88842ebcac9f5542d51e89eb350d2f6da78407dab7bb5c01bb8860c32c9eef4c (manifest had different hash; current NCBI file used)
- GSE216211_RAW.tar downloaded; SHA-256: 4ad865a6c291ecc1c90b14f084b6dbf5081c892a25110b2261a7f9956c09232a
- Pseudobulk (sum counts per biological sample) → log1p(CPM) → top-2000 variance genes
- GSE153480 prepared table: 8 samples (4 MI / 4 Sham), 2000 features, fingerprint b6a6c04d0910111b
- GSE216211 prepared: 2 samples (Sham, MI; MI-E excluded). Too few biological groups for group-aware train/val/test split (requires ≥3). External evaluation deferred.

## Split (GSE153480, group-aware)
- n_train=4, n_validation=2, n_test=2
- train labels: {0: 2, 1: 2}
- val labels: {1: 2}
- test labels: {0: 2}

## Model matrix (validation metrics; test evaluated once without tuning)

### logistic_regression
- validation: {'accuracy': 1.0, 'balanced_accuracy': 1.0, 'f1_macro': 1.0, 'auroc': nan}
- held-out test (once, no tuning): {'accuracy': 1.0, 'balanced_accuracy': 1.0, 'f1_macro': 1.0, 'auroc': nan}

### hist_gradient_boosting
- validation: {'accuracy': 0.0, 'balanced_accuracy': 0.0, 'f1_macro': 0.0, 'auroc': nan}
- held-out test (once, no tuning): {'accuracy': 1.0, 'balanced_accuracy': 1.0, 'f1_macro': 1.0, 'auroc': nan}

## Unit / invariant tests
- `pytest tests/`: **22 passed** (scientific invariants, splitting, benchmarks, integrity, multimodal, training)

## Limitations (honest)
- Sample size is very small (n=8 biological samples). Metrics are unstable; single-class validation folds can occur depending on split.
- AUROC undefined when validation/test has only one class.
- Feature selection (top variance genes) was performed on the full GSE153480 table before split (leakage relative to ideal fold-wise selection). For production, move HVGs inside training folds only.
- Gene spaces between GSE153480 and GSE216211 were not aligned; cross-study transfer not executed.
- Manifest SHA-256 values in repo do not match current NCBI downloads; manifests should be updated.
- MLP baseline failed on this tiny n due to internal stratified validation requirements.

## Conclusion
Pipeline runs end-to-end on real public GEO data after local materialization. Group-aware splitting and model matrix execute. Results are not scientifically publishable due to n and residual leakage in HVG selection; they demonstrate that the code path is functional.
