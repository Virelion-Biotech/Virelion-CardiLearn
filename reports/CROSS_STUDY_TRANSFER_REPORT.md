# Cross-study MI-vs-Sham transfer (next step)

Date: 2026-08-30 17:06 UTC
Protocol: shared gene intersection → train-only top-2000 HVG → model → external study as pure test
No test-set leakage in feature selection.

Shared genes before HVG: 20529
GSE153485 n=20 labels={1: 10, 0: 10}
GSE236374 n=9 labels={1: 6, 0: 3}

## Train GSE153485 → Test GSE236374
- HVG count (train-only): 2000
- **logistic_regression**: {'accuracy': 1.0, 'balanced_accuracy': 1.0, 'f1_macro': 1.0, 'auroc': 1.0, 'auprc': 1.0}
- **hist_gradient_boosting**: {'accuracy': 0.3333333333333333, 'balanced_accuracy': 0.5, 'f1_macro': 0.25, 'auroc': 0.5, 'auprc': 0.6666666666666666}

## Train GSE236374 → Test GSE153485
- HVG count (train-only): 2000
- **logistic_regression**: {'accuracy': 0.5, 'balanced_accuracy': 0.5, 'f1_macro': 0.3333333333333333, 'auroc': 0.85, 'auprc': 0.8708180708180707}
- **hist_gradient_boosting**: {'accuracy': 0.5, 'balanced_accuracy': 0.5, 'f1_macro': 0.3333333333333333, 'auroc': 0.5, 'auprc': 0.5}

## Within GSE153485 (train-only HVG, held-out 30% stratified)
- **logistic_regression**: {'accuracy': 1.0, 'balanced_accuracy': 1.0, 'f1_macro': 1.0, 'auroc': 1.0, 'auprc': 1.0}
- **hist_gradient_boosting**: {'accuracy': 0.5, 'balanced_accuracy': 0.5, 'f1_macro': 0.3333333333333333, 'auroc': 0.5, 'auprc': 0.5}

## Interpretation
- Cross-study transfer tests whether MI vs Sham signal generalizes across bulk cohorts with shared genes and no test leakage in HVG selection.
- Within-study held-out is a cleaner baseline than previous runs that selected HVGs on the full table.
- Scores on small n can still be fragile; larger multi-study matrices remain the next scientific step.
