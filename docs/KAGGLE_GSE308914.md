# Kaggle GSE308914 analysis

`scripts/kaggle_gse308914_full_pipeline.py` is the canonical notebook/script entry point for the GSE308914 exploratory validation wave.

## Scientific target

The analysis asks whether sample-level expression distinguishes **D0 baseline/reference from post-MI** samples. D0 is explicitly **not surgical sham**.

## Leakage control

Every resampling split independently learns:

1. sample-wise normalization parameters;
2. log transformation;
3. highly variable feature ranking and selection;
4. imputation;
5. scaling;
6. model parameters.

The permutation analysis repeats the complete fold-wise pipeline after label permutation. This prevents the previous full-dataset HVG-selection shortcut from contaminating validation folds.

## Kaggle execution

1. Enable Internet in the Kaggle notebook.
2. Clone or upload this repository.
3. Run:

```bash
python scripts/kaggle_gse308914_full_pipeline.py \
  --out /kaggle/working/cardilearn_gse308914 \
  --folds 5 \
  --repeats 10 \
  --permutations 100 \
  --hvg-k 2000
```

If GEO download succeeds but the expression matrix is not automatically discoverable, place the processed expression matrix in the raw directory and rerun with `--matrix /path/to/matrix.csv`.

## Outputs

- `tables/sample_metadata.csv` — parsed GEO sample metadata and labels
- `tables/aligned_expression.csv` — expression matrix after sample alignment
- `tables/nested_cv_fold_metrics.csv` — fold-level metrics
- `tables/nested_cv_summary.csv` — aggregate metrics
- `tables/oof_predictions.csv` — out-of-fold probabilities
- `tables/permutation_test.json` — complete-pipeline permutation null
- `tables/feature_selection_frequency.json` — fold-wise feature-selection frequency
- `figures/oof_roc.png` — out-of-fold ROC curve
- `figures/permutation_null.png` — permutation null distribution
- `run_manifest.json` — parameters, input hash, cohort counts and headline results

## Interpretation guardrail

A high AUROC here establishes discrimination within this cohort under the specified resampling scheme. It does **not** establish sham-vs-MI performance, clinical utility, causal biology, or cross-study generalization. Those require independent datasets and separately locked validation.