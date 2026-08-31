# GSE308914 leakage-safe reanalysis protocol

This protocol supersedes the earlier exploratory GSE308914 result that selected approximately 2,000 HVGs before cross-validation.

## What the canonical Kaggle run now does

1. Downloads the GEO family SOFT record from NCBI.
2. Downloads the individual 10x Matrix Market supplementary files for the 30 GSM samples.
3. Parses sample-level timepoint and sex metadata.
4. Aggregates cells within each biological sample to one all-cell pseudobulk count vector.
5. Treats the biological sample—not individual cells—as the inferential unit.
6. Performs repeated stratified cross-validation.
7. Inside every training fold, independently performs library-size normalization, log1p transformation, HVG selection, imputation, scaling, and logistic-regression fitting.
8. Runs leave-one-sex-out transfer tests.
9. Runs a full-pipeline permutation null in which labels are shuffled before fold-specific feature selection.
10. Records feature-selection frequency, sample-level QC, out-of-fold metrics, permutation statistics, and a run manifest.

## Interpretation

The primary endpoint is **D0 baseline/reference vs post-MI discrimination**. D0 is not surgical sham and must not be reported as sham.

A high score on this cohort does not establish external generalization. It may reflect real injury biology, cell-composition shifts, experimental design, or other cohort-specific signal. The analysis therefore reports sample-level QC and treats cross-study validation as a separate requirement.

## Kaggle

With Internet enabled:

```bash
python scripts/kaggle_gse308914_full_pipeline.py \
  --out /kaggle/working/cardilearn_gse308914 \
  --folds 5 \
  --repeats 10 \
  --hvg-k 2000 \
  --permutations 100
```

For a fast smoke test before a full run, use `--repeats 1 --permutations 5` and then rerun the locked analysis with the defaults.

## Output package

`VALIDATION_REPORT.md`, `run_manifest.json`, sample metadata/QC tables, aligned pseudobulk counts, repeated-CV fold metrics, CV summary, out-of-fold-derived metrics, feature-selection frequency, leave-one-sex-out results, leave-one-timepoint-out results when estimable, and the full permutation null are written beneath the requested output directory.

The raw GEO files are intentionally written outside Git and are hashed/represented in the local run manifest rather than committed to the repository.