# GSE153480 — biological review record

## Status

**Candidate / not approved for data lock.**

This record is intentionally conservative. The GEO series is suitable as a candidate single-cell RNA dataset, but the repository must not treat each GEO sample accession as an independent biological subject without explicit evidence.

## Source evidence

- GEO series: `GSE153480`
- BioProject: `PRJNA642814`
- Organism: `Mus musculus`
- Assay: single-cell RNA-seq
- Platform: Illumina NextSeq 500 / GPL19057
- Samples: 8
- Processed data: MTX/TSV supplementary files
- GEO states that P1 or P8 mice underwent MI or sham surgery and ventricle tissue below the surgery plane was collected at 1 or 3 days after surgery.

## Samples listed by GEO

| GSM | Condition encoded by GEO title | Developmental age | Post-surgical day | Surgery |
|---|---|---:|---:|---|
| GSM4644949 | scRNA-seq_P1_1MI | P1 | 1 | MI |
| GSM4644950 | scRNA-seq_P1_1Sham | P1 | 1 | sham |
| GSM4644951 | scRNA-seq_P1_3MI | P1 | 3 | MI |
| GSM4644952 | scRNA-seq_P1_3Sham | P1 | 3 | sham |
| GSM4644953 | scRNA-seq_P8_1MI | P8 | 1 | MI |
| GSM4644954 | scRNA-seq_P8_1Sham | P8 | 1 | sham |
| GSM4644955 | scRNA-seq_P8_3MI | P8 | 3 | MI |
| GSM4644956 | scRNA-seq_P8_3Sham | P8 | 3 | sham |

## What is explicitly supported

The accession-level condition, developmental age, collection timepoint, tissue, organism, and 10x scRNA-seq modality are supported by the GEO records.

## What remains unresolved

### 1. Biological subject hierarchy

The GEO sample page identifies the sample accession and experimental condition, but this review record does **not** establish whether each GSM corresponds to one mouse, pooled mice, or another biological sampling unit. Therefore:

- `subject_id` is unresolved.
- `subject_confidence` must remain `unverified`.
- A GSM accession must not automatically be promoted to `subject_id` merely because it is a distinct GEO sample.

### 2. Biological replicate count

The 8 GEO samples are not sufficient evidence by themselves to claim eight independent biological replicates. Replicate structure must be reconciled against the original methods/source data before the dataset can contribute to a locked biological split.

### 3. Cell-level labels

The GEO record supplies the expression matrix and experimental metadata, but final cell-type labels must be taken from an authoritative processed annotation/source rather than inferred from gene expression during the lock stage.

### 4. QC provenance

GEO states that Seurat-based downstream processing removed potential doublets, red blood cells, and low-quality cells. CardiLearn should nevertheless perform its own structural/QC checks on the downloaded processed matrices and retain the source processing information as provenance rather than silently treating the GEO filtering as CardiLearn QC.

## Proposed canonical labels after review

These are **candidate mappings**, not locked labels:

- `species = Mus musculus`
- `assay = scRNA-seq`
- `tissue = ventricle`
- `injury = myocardial_injury` for MI samples
- `injury = reference` for sham samples
- `maturation = P1` or `P8` as an ordinal/developmental state, subject to the project's canonical maturation ontology
- `timepoint = 1d` or `3d`
- `study_id = GSE153480`
- `study_family_id = wang_olson_neonatal_regeneration`

The `injury` and `maturation` mappings must not be confused with a claim that all P1 samples are biologically equivalent or that P8 is intrinsically a generic 'non-regenerative' label. Those are experimental states that require task-specific interpretation.

## Lock decision

**NOT ELIGIBLE FOR LOCK YET.**

Required before approval:

1. Establish subject/pooling structure from the primary paper and/or authoritative supplementary metadata.
2. Establish biological replicate identity and independence.
3. Confirm the exact downloaded matrix files and SHA-256 hashes.
4. Reconcile cell-level annotation provenance.
5. Run structural/QC checks after acquisition.
6. Confirm that the study-family assignment does not duplicate another candidate source from the same biological experiment.

## Acquisition policy

Use the processed GEO MTX/TSV files. Do **not** download the 136 GB SRA raw sequencing data for this training cohort merely to reproduce the original sequencing pipeline. Raw data remain unnecessary for the current CardiLearn representation-learning objective unless a later analysis specifically requires raw reprocessing.

## Review principle

This study can be highly valuable for CardiLearn because it directly spans regenerative P1 and less-regenerative P8 cardiac injury contexts, but that biological usefulness does not override the independence/leakage requirements of the training split.
