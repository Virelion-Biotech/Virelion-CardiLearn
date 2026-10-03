# Step 19 — Cohort Eligibility Review

This document defines the gate between public dataset discovery and actual CardiLearn training.

## Current state

**No dataset is training-approved.** The repository remains in `candidate_sources_not_locked` state.

The cohort matrix in `configs/cohort_eligibility_v1.yaml` records candidate roles and blockers. It is intentionally conservative: GEO sample counts are not treated as biological replicate counts, and no label is inferred from filenames alone.

## Evidence hierarchy

For each candidate, resolve:

`study_family_id → study_id → subject_id → sample_id → cell_or_nucleus_id`

A partition may not contain the same biological subject, donor, or animal in more than one of train/validation/test. Cells or nuclei from the same biological sample are not independent replicates.

## First acquisition target: GSE153480

GSE153480 contains eight scRNA-seq samples covering P1/P8, MI/sham, and 1/3-day post-surgery states. NCBI provides processed MTX/TSV data, making this preferable to downloading the much larger SRA data for the first assembly pass.

Before approval we must verify:

1. exact sample-to-BioSample relationships;
2. animal/subject identity and biological replicate structure;
3. tissue/region semantics;
4. MI versus sham and P1 versus P8 labels;
5. post-surgery timepoint;
6. cell-type annotation provenance;
7. cell-level QC distributions;
8. sample-level cell counts and QC outliers;
9. gene identifier format and reference genome/annotation;
10. study-family relationship to related Wang/Olson datasets;
11. whether the eight GEO samples represent eight independent biological subjects or repeated sampling/other structure.

The scATAC companion series is excluded from the RNA matrix.

## Cross-species policy

GSE185289 is a high-value mammalian regeneration candidate, but its 56 GEO samples must not be interpreted as 56 independent animals. The source reports multiple pig experimental groups and developmental states; animal-level metadata must be reconciled before split assignment.

GSE217494 is a high-value human external candidate. Its design describes 22 explanted hearts across healthy, acute MI, chronic ischemic, and non-ischemic cardiomyopathy groups. The RNA/GEX branch is eligible for consideration; CITE/ADT measurements remain separate unless CardiLearn receives an explicit multimodal branch.

Zebrafish scRNA-seq and Stereo-seq are separate modalities. Orthology mappings are explicit, deterministic, and frozen before cross-species integration.

## QC gate

Matrix-level QC must detect at minimum:

- duplicate observation IDs;
- zero/near-zero observations;
- non-finite expression values;
- impossible matrix dimensions;
- inconsistent gene identifiers;
- duplicate genes after mapping;
- sample-to-subject ambiguity;
- subject-to-study ambiguity;
- missing required labels;
- sample-level low-cell-count outliers;
- extreme library-size and detected-feature distributions;
- mitochondrial/ribosomal fractions where supported by the source/reference annotation;
- suspicious batch/sample effects requiring explicit review.

QC thresholds must be documented before being used to exclude observations. They must not be tuned using the locked test partition.

## Approval gate

A source can move from `candidate` to `approved` only after:

- source metadata are archived with provenance and hashes;
- subject/sample hierarchy is verified;
- study-family mapping is reviewed;
- matrix and metadata pass structural QC;
- canonical labels are resolved;
- gene mapping is frozen;
- a reviewed split manifest exists;
- the split covers the required independent study families;
- no biological unit crosses partitions;
- the resulting lock manifest is fingerprinted.

Until those conditions are satisfied, CardiLearn may run only acquisition, reconciliation, QC, and synthetic/software tests—not empirical biological training claims.
