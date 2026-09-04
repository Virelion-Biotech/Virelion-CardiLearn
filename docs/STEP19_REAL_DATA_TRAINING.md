# Step 19 — Real-data training

Step 19 is the transition from a tested training framework to an actual biological dataset. It is intentionally a **data-lock and execution gate**, not permission to manufacture benchmark results.

## Required sequence

1. Materialize source metadata and expression outside Git.
2. Reconcile study, subject, sample/library, cell/nucleus, condition, timepoint, tissue/region, modality, and technical-replicate relationships.
3. Resolve the biological labels required by the prototype (`maturation`, `injury`, `cell_type`). Unknown labels are rejected rather than guessed.
4. Freeze a study-grouped split before model fitting.
5. Fingerprint the matrix, metadata, and split manifest.
6. Fit gene selection and any normalization/imputation parameters on training observations only.
7. Train CardiLearn using only the training partition.
8. Preserve validation/test rows strictly for downstream evaluation.
9. Run the locked Step 15 benchmark and cross-study/species evaluation before making performance claims.

## Training-ready input contract

Expression must be observations × genes and can be supplied as `.npz` (`X`, `genes`), `.csv`, or `.tsv`. Metadata can be CSV or Parquet and must contain:

- `study_id`
- `subject_id`
- `sample_id`
- `species`
- `assay`
- `cell_type`
- `maturation`
- `injury`

`study_id` is the default independence unit for the initial real-data runner. If related accessions belong to the same biological study family, a `study_family_id` should replace it before the lock.

## Execution

```bash
pip install -e '.[torch]'
python scripts/train_real_data.py \
  --expression /path/to/training_ready_expression.npz \
  --metadata /path/to/training_ready_metadata.parquet \
  --output runs/real-data-v1 \
  --seed 42
```

The runner automatically uses CUDA when available and otherwise CPU. It does not commit raw expression data or generated model artifacts to Git.

## Current scientific status

The repository's candidate GEO studies are **not yet a locked training corpus**. Their accession-level metadata establish candidates, not independently verified sample-level labels. Training therefore remains pending completion of the metadata reconciliation/data-lock step.

A successful training run means that optimization completed. It does **not** establish that CardiLearn learns conserved cardiac biology, outperforms baselines, predicts regeneration, or has clinical utility.
