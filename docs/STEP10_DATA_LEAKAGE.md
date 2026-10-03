# CardiLearn Step 10 — Data Leakage Control

Step 10 is a hard scientific gate, not a reporting preference. No result is considered part of a publishable benchmark until the relevant leakage checks pass.

## Hierarchy invariant

The protected biological hierarchy is:

```text
study family → study → subject/animal/donor → sample/library → cell/nucleus
```

The primary independence unit for cross-study evaluation is `study_family_id`. Within a study, biological subjects remain indivisible. Samples and cells derived from the same subject cannot cross partitions.

Repeated `sample_id` values are expected in cell/nucleus-level tables. The invariant is that one sample maps to one partition.

## Hard leakage gates

`cardilearn.leakage.audit_hierarchy()` blocks:

- missing study, subject, or sample identifiers;
- a study family appearing in multiple partitions;
- a study appearing in multiple partitions;
- a subject appearing in multiple partitions;
- a sample appearing in multiple partitions;
- missing or unknown train/validation/test partition labels.

The same checks are run before the existing train-only gene-selection function is allowed to inspect expression variance.

## Frozen split manifest

`freeze_split_manifest()` records the assignment of every independent group and computes a SHA-256 hash over the canonical assignment payload. `verify_frozen_split_manifest()` detects reassignment or manifest tampering.

This makes the benchmark split reproducible without storing expression matrices in GitHub.

## Feature leakage

Feature selection must be fitted on training observations only. The implementation therefore requires:

1. a completed partition assignment;
2. no hierarchy leakage;
3. variance selection using only `_split == train` rows;
4. selected gene indices saved with the split/training record.

The validation and test sets must not influence HVG selection, normalization parameters, imputation parameters, scaling parameters, dimensionality reduction fit, threshold selection, or model architecture selection.

## Exact expression duplicate check

`exact_cross_split_feature_collisions()` is deliberately diagnostic rather than a hard failure. Sparse scRNA-seq/snRNA-seq data can produce identical count vectors in biologically unrelated cells, so exact equality alone is not sufficient evidence of leakage. Any collision should be reviewed against subject/sample metadata.

## Test-set protection

The test partition is treated as an external evaluation set. Hyperparameter selection, architecture selection, feature selection, stopping criteria, threshold optimization, and model selection must use training/validation data only.

The final test set should be read exactly once for the primary claim after the split manifest is frozen.

## Related-series protection

Separate GEO accessions are not assumed to be independent studies. The versioned study-family registry groups linked series before external evaluation. The current registry explicitly groups GSE130699 and GSE153480 into one neonatal regeneration family.

## What still remains before Step 10 is complete

The code-level protections are implemented, but the real benchmark is not yet locked. Sample-level biological reconciliation must establish subject/replicate mappings for the candidate datasets, followed by a reviewed family-level split and a frozen manifest. Only then can expression matrices be materialized for the scientific benchmark.
