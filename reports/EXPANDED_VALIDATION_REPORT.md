# Expanded MI-vs-Sham Validation Report

Date: 2026-08-30
Additional data: GSE236374 (bulk RNA-seq, 9 biological samples: 3 Sham + 6 MI at 7d/28d)

### GSE153480
- n=8, labels={1: 4, 0: 4}, fingerprint=b6a6c04d0910111b
- split: train=4 {0: 2, 1: 2}, val=2 {1: 2}, test=2 {0: 2}
- logistic_regression: val={'accuracy': 1.0, 'balanced_accuracy': 1.0, 'f1_macro': 1.0, 'auroc': nan} | test={'accuracy': 1.0, 'balanced_accuracy': 1.0, 'f1_macro': 1.0, 'auroc': nan}
- hist_gradient_boosting: val={'accuracy': 0.0, 'balanced_accuracy': 0.0, 'f1_macro': 0.0, 'auroc': nan} | test={'accuracy': 1.0, 'balanced_accuracy': 1.0, 'f1_macro': 1.0, 'auroc': nan}

### GSE236374
- n=9, labels={1: 6, 0: 3}, fingerprint=571abc04d46f2010
- split: train=5 {1: 3, 0: 2}, val=2 {1: 2}, test=2 {0: 1, 1: 1}
- logistic_regression: val={'accuracy': 0.5, 'balanced_accuracy': 0.5, 'f1_macro': 0.3333333333333333, 'auroc': nan} | test={'accuracy': 1.0, 'balanced_accuracy': 1.0, 'f1_macro': 1.0, 'auroc': 1.0}
- hist_gradient_boosting: val={'accuracy': 1.0, 'balanced_accuracy': 1.0, 'f1_macro': 1.0, 'auroc': nan} | test={'accuracy': 0.5, 'balanced_accuracy': 0.5, 'f1_macro': 0.3333333333333333, 'auroc': 0.5}

## Unit tests
- pytest tests/: 22 passed (prior run)

## Additional datasets researched / status
- GSE163465: downloaded + extracted (CD45+ Sham + MI Day3/7/14). Only 1 Sham biological sample → insufficient for group-aware balanced binary splits; not included in matrix.
- GSE135310: listed in CardiBench sources; RAW.tar ~2.1 GB — too large for this runtime environment; deferred.
- GSE269054: spatial/Visium (priority 3 in matrix YAML); deferred for modality alignment.
- GSE106472 / GSE216211: already considered; GSE216211 only 2 usable groups after MI-E exclusion.
- Gene spaces remain study-specific (top-2000 HVGs per study). True multi-study transfer needs shared vocabulary + batch handling.

## Conclusion
Added independent bulk RNA-seq cohort GSE236374 (n=9, 3 Sham + 6 MI). Pipeline executes on both scRNA-seq pseudobulk and bulk. Overall biological n remains modest; metrics still sensitive to single-class partitions. Code path validated on real public data.
