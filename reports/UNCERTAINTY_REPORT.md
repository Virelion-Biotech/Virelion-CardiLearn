# Uncertainty quantification (permutation + bootstrap)

Date: 2026-08-30T17:26:21.728516+00:00
Target: logistic GSE153485 → GSE236374 (best cross-study direction)

- Observed AUROC: **1.000**, balanced accuracy: **1.000**
- Permutation null (n=200, shuffle train labels): mean AUROC=0.471, one-sided p=0.3582
- Bootstrap test resample (n_valid=483): 95% CI AUROC [1.000, 1.000]

## Interpretation
- One-sided permutation p≈0.36 means AUROC=1.0 is **not rare** under a null label shuffle when the external test set has only n=9 samples.
- Bootstrap CI collapsing to [1, 1] reflects that every valid resample of this tiny test set also scored perfectly — not strong evidence of stability at larger n.
- Do not treat the point estimate of 1.0 as a definitive biological or ML claim.

## Stop point
No further high-value steps remain in this environment without larger multi-study matrices or longer compute.
Prior work already covers materialization, within-cohort matrices, cross-study shared-gene transfer, definition fixes, and reports on main.
