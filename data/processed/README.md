# Processed sample features

## Cohorts in this validation wave

| File | Study | n | Labels |
|------|-------|---|--------|
| `GSE308914_sample_meta.csv` | GSE308914 | 30 | 6 D0 baseline + 24 post-MI |
| `GSE135310_sample_meta.csv` | GSE135310 | 5 | 1 steady-state + 4 MI days |

Full feature matrices (`*_sample_features.csv`, ~2000 HVG columns) are produced in the Kaggle/local pipeline and are **not** stored in git (size + anonymous gene rank columns). Metrics and the strong-evaluation report live under `reports/` and `runs/mi-sham-matrix/`.

See `reports/GSE308914_VALIDATION_REPORT.md` for results (CV AUROC 1.0, leave-one-sex-out, permutation p≈0.005).

**Label note:** GSE308914 D0 is baseline, not surgical sham.
