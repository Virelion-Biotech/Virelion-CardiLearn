# Step 19 — NCBI data acquisition and reconciliation

The repository now contains a metadata-only acquisition utility:

```bash
python scripts/fetch_ncbi_metadata.py --output data/ncbi_metadata
```

It fetches public metadata/provenance for the current candidate sources without downloading or committing expression matrices:

- `GSE185289` — Nakada pig snRNA-seq
- `PRJNA1233465` — Gao zebrafish regeneration atlas
- `GSE217494` — human MI/fibrosis CITE-seq

## Why metadata comes first

The three sources are biologically complementary but are not automatically mergeable. GSE185289 contains 56 GEO samples and multiple pig injury/development groups; the published study reports 218,945 high-quality nuclei overall and 94,844 cardiomyocytes. PRJNA1233465 contains scRNA-seq plus Stereo-seq across eight regeneration stages. GSE217494 contains CITE-seq from 22 explanted human hearts. These differences make sample hierarchy, assay, tissue, condition, and gene-identifier reconciliation mandatory before a matrix is locked.

## Required reconciliation fields

For every expression observation, the final manifest must identify:

`study_family_id → study_id → subject_id → sample_id → cell_or_nucleus_id`

and normalized values for:

- species
- assay/modality
- tissue/region
- cell type
- condition
- injury state
- maturation state
- timepoint
- biological replicate / donor where applicable

Technical replicates must not become independent biological units.

## Species policy

For the first cross-species model, retain only genes passing the frozen orthology policy. The default policy should be conservative one-to-one orthologs with stable identifiers. Species-specific genes can be retained in a separate species-private branch but must not silently enter the shared latent-space feature matrix.

## Important exclusions

- Stereo-seq is not silently merged with scRNA-seq.
- CITE-seq protein features are not silently merged with RNA expression.
- Cell-level random splitting is prohibited.
- Unknown maturation/injury/cell-type labels are not guessed.
- Raw FASTQ/SRA/GEO archives are not committed to Git.
- A candidate remains a candidate if any required reconciliation check fails.

## Current status

This acquisition layer verifies that the public sources are accessible and records their metadata provenance. It does **not** constitute a data lock. A true lock requires materialized expression, complete sample/cell metadata, orthology mapping, normalized labels, hierarchy checks, split fingerprint, and expression/metadata fingerprints.
