# Full multi-cohort MI-vs-Sham validation (tried here)

Date: 2026-08-30

### GSE153480
- n=8, labels={1: 4, 0: 4}, fingerprint=b6a6c04d0910111b
- split: train=4 {0: 2, 1: 2}, val=2 {1: 2}, test=2 {0: 2}
- logistic_regression: val AUROC nan, bal_acc 1.0 | test AUROC nan, bal_acc 1.0
- hist_gradient_boosting: val bal_acc 0.0 | test bal_acc 1.0

### GSE236374
- n=9, labels={1: 6, 0: 3}, fingerprint=571abc04d46f2010
- split: train=5, val=2, test=2
- logistic_regression: val bal_acc 0.5 | test AUROC 1.0, bal_acc 1.0
- hist_gradient_boosting: val bal_acc 1.0 | test AUROC 0.5, bal_acc 0.5

### GSE153485 (NEW — primary)
- n=20, labels={1: 10, 0: 10}, fingerprint=1980f79bab42d3f1
- split: train=12 (6/6), val=4 (2/2), test=4 (2/2)
- **logistic_regression: val AUROC 1.0, bal_acc 1.0 | test AUROC 1.0, bal_acc 1.0**
- hist_gradient_boosting: val AUROC 0.5, bal_acc 0.5 | test AUROC 0.5, bal_acc 0.5

### GSE106472
- SKIPPED: only 2 biological groups (need ≥3)

## Cohorts attempted this session
- **GSE153485** (bulk, heart only): 20 samples, 10 MI / 10 Sham — primary new cohort
- GSE106472: WT NOMI vs D4MI only 2 groups — skipped for group-aware split
- GSE135310: soft downloaded; RAW 2.1G contains multiplexed samples; steady-state labeled under MI condition in soft — selective mtx download possible but control design ambiguous; deferred full matrix
- GSE308914: 4.9G RAW — too large for practical processing here

## Conclusion
Largest clean cohort is GSE153485 (n=20 balanced bulk heart MI vs Sham). Logistic regression separates cleanly under group-aware splits. HGB underperforms on this small-n table. Metrics remain exploratory (per-study HVG selection; no cross-study gene alignment yet). Pipeline validated on real public GEO data.
