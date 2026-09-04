# Step 19 — Real-data metadata reconciliation

CardiLearn now has a conservative GEO sample reconciliation stage between NCBI metadata acquisition and cohort assembly.

## Purpose

`python scripts/reconcile_geo_samples.py` converts a GEO family-SOFT file into a reviewable sample-level CSV. It does **not** inspect expression matrices, infer labels from filenames, or approve biological metadata.

The source hierarchy is preserved as:

`study_family_id → study_id → subject_id → sample_id → cell_or_nucleus_id`

The reconciliation script operates at the `study_id → subject_id → sample_id` level. Cell/nucleus identifiers are attached later by the 10x assembler.

## Conservative behavior

- Subject identifiers are accepted only when explicitly present in GEO characteristics.
- Condition mapping is deliberately narrow (`sham/control/healthy → reference`; `MI/infarct/myocardial injury → myocardial_injury`).
- Unknown conditions remain `unresolved`.
- Missing maturation, injury, cell type, tissue, region, and timepoint labels remain empty/unresolved unless explicitly present.
- A source-explicit subject is marked `source_explicit_unreviewed`, never `verified`.
- Every output row receives review issues.
- The output cannot by itself satisfy the real-data lock.

## Why this matters

GEO processed single-cell data are expected to retain cell-level identifiers and sample-level relationships. NCBI's single-cell submission guidance explicitly describes per-sample MEX/HDF5/RDS processed data and requires the sample metadata to identify the corresponding data files. citeturn0search1

CardiLearn therefore separates:

1. **Acquisition** — obtain authoritative metadata and processed files.
2. **Reconciliation** — expose source-provided relationships without guessing.
3. **Human review** — verify subject/sample semantics, biological labels, modality and study-family independence.
4. **Assembly** — construct a sparse cell-level matrix using the approved sample manifest.
5. **Lock** — fingerprint expression, metadata, genes and family-level split.
6. **Training** — only after the lock passes.

## Example

```bash
python scripts/reconcile_geo_samples.py \
  --soft /data/cardilearn/ncbi/GSE153480_family.soft.gz \
  --study-id wang_neonatal_regeneration_2020 \
  --accession GSE153480 \
  --species 'Mus musculus' \
  --assay scRNA-seq \
  --output /data/cardilearn/reconciled/GSE153480_samples.csv \
  --summary /data/cardilearn/reconciled/GSE153480_audit.json
```

The resulting CSV is a **review artifact**, not a training manifest. Before copying rows into `configs/real_data_sample_manifest_v1.csv`, a reviewer must verify the accession, subject identity, sample relationships, biological labels, study-family assignment, and modality against the authoritative source records.
