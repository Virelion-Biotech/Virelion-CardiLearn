# GSE308914 + GSE135310 sample-level validation

**Date:** 2026-08-31  
**Repo:** Virelion-Biotech/Virelion-CardiLearn  
**Task:** MI / post-injury vs reference on sample pseudobulk (scRNA → sum → log1p CPM → top-2000 HVG)

## Data

| Study | n samples | Reference | Injury | Notes |
|-------|-----------|-----------|--------|-------|
| **GSE308914** | 30 | 6 (D0 baseline) | 24 (D1, D4, D7, D28) | Primary result |
| GSE135310 | 5 | 1 (steady state F8) | 4 (MI days) | Too small for stable ML; footnote only |

Files:
- `data/processed/GSE308914_sample_features.csv`
- `data/processed/GSE135310_sample_features.csv`
- `runs/mi-sham-matrix/GSE308914_STRONG_EVAL.json`
- `runs/mi-sham-matrix/gse308914_within_study_metrics.csv`

**Label note:** GSE308914 uses **day-0 baseline vs post-MI**, not classic surgical sham.

## GSE308914 — strongest evaluation (logistic, class-balanced)

### Stratified 5-fold CV
- AUROC mean = **1.000** (folds: [1.0, 1.0, 1.0, 1.0, 1.0])

### Leave-one-sex-out
- Test sex **F** (n_test=15): AUROC=1.000, balanced accuracy=0.958
- Test sex **M** (n_test=15): AUROC=1.000, balanced accuracy=1.000

### Timepoint contrasts (D0 vs each post-MI day)
- **D0_vs_D1**: mean AUROC=1.000
- **D0_vs_D4**: mean AUROC=1.000
- **D0_vs_D7**: mean AUROC=1.000
- **D0_vs_D28**: mean AUROC=1.000

### Permutation null (label shuffle)
- n_perm = 200
- Observed CV AUROC = **1.000**
- Null mean AUROC = 0.497, null max = 0.960
- One-sided p ≈ **0.0050**

## Holdout snapshot (earlier single split)

See `runs/mi-sham-matrix/gse308914_within_study_metrics.csv`:
- logistic_regression: AUROC 1.0 on n_test=9
- hist_gradient_boosting: AUROC 0.5 (majority-class collapse; not informative at this n)

## GSE135310

n=5 (1 reference). Leave-one-out only partially scorable (single-class train folds when the sole reference is held out). **Not used as a primary claim.**

## Honest limitations
1. Only **6** reference samples in GSE308914.
2. D0 baseline ≠ surgical sham.
3. HVG selected on the full table before CV → mild optimistic bias possible.
4. No external independent cohort in this commit.
5. Gene columns are anonymous `gene_0…gene_1999` (not symbols) — true cross-study gene alignment not possible from these CSVs alone.

## Claim (careful)
On GSE308914 sample-level pseudobulk (n=30), balanced logistic regression separates day-0 baseline from post-MI with AUROC 1.0 under stratified CV, leave-one-sex-out, and each day contrast, with permutation p≈0.005. This supports a strong linear baseline-vs-injury signal in this cohort; external validation and train-fold-only HVG remain recommended before broader product claims.
