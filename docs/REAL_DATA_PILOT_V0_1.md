# CardiLearn v0.1 — Real-data pilot

## Purpose

This is the first real-data integration layer for the new CardiLearn research model. It is deliberately **candidate-only**: an accession being listed here does not mean that its expression matrix has been downloaded, its biological subject mapping has been reconciled, or that it is safe to use as a locked benchmark.

The code/data boundary is:

```text
CardiAtlas / CardiBench
        ↓
canonical study + sample metadata
        ↓
CardiLearn real-data audit
        ↓
feature/task eligibility
        ↓
locked split
        ↓
training matrix
```

Source expression data are not committed to GitHub. `scripts/materialize_geo.py` performs explicit GEO acquisition and records a SHA-256 checksum.

## Pilot studies

### GSE185289 — pig regeneration/development/injury

Species: *Sus scrofa*  
Modality: single-nucleus RNA-seq  
Primary role: developmental maturation, injury, regeneration, and state ordering.

The GEO record describes fetal and postnatal control states, MI, apical resection, and combined apical-resection/MI groups across P1–P56. The accompanying publication reports cardiomyocyte and broader cardiac single-nucleus profiles and explicitly studies the extension of the regenerative window.

This is the strongest initial pilot source for regeneration-related relational supervision, but subject/replicate mappings still have to be recovered from sample metadata before constructing training pairs.

Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE185289

### GSE130699 — mouse neonatal injury/regeneration

Species: *Mus musculus*  
Modality: single-nucleus RNA-seq  
Primary role: maturation, injury, regenerative-vs-non-regenerative developmental context, and temporal ordering.

The GEO record contains P1/P8 MI and sham groups at one and three days after surgery. The study therefore provides a compact cross-species counterpart to the pig regenerative dataset.

Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130699

### GSE217494 — human myocardial-injury cellular context

Species: *Homo sapiens*  
Modality: CITE-seq/GEX  
Primary role: human injury/state representation and external cell-context validation.

The GEO record describes 22 explanted human hearts spanning healthy donors, acute MI, chronic ischemic cardiomyopathy, and non-ischemic cardiomyopathy. This cohort should primarily supervise/validate injury and cellular-context representations. It should **not** be used as a maturation or regeneration ground truth merely because the samples are clinically different.

Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE217494

## Task eligibility

| Study | Representation | Maturation | Injury | Regeneration | Transition |
|---|---:|---:|---:|---:|---:|
| GSE185289 | yes | candidate | candidate | candidate, high priority | candidate |
| GSE130699 | yes | candidate | candidate | candidate, high priority | candidate |
| GSE217494 | yes | no | yes | no | no |

`candidate` means the study's design is compatible with the task, not that the labels have already been verified at sample level.

## Required canonical sample metadata

A recovered sample table must contain at least:

```text
study_id
subject_id
sample_id
condition
timepoint
modality
species
tissue
region
cell_context
replicate_group
is_technical_replicate
```

Unknown values remain unknown. Missing subject identifiers are a blocking problem for primary biological inference.

## Required hierarchy

```text
study → subject → sample → cell
```

Technical replicates remain attached to their biological parent. Samples from the same animal/donor must never cross train/validation/test partitions.

## Lock criteria

A dataset may enter a locked CardiLearn benchmark only after all of the following are true:

1. GEO/ENA accession identity is verified.
2. Sample metadata are materialized and checksum-recorded.
3. Subject/donor/animal identifiers are recovered or an explicit weaker split policy is approved.
4. Technical replicate grouping is resolved.
5. Condition and timepoint labels are normalized without ambiguous guessing.
6. Modality and tissue are verified.
7. The study passes hierarchy-leakage checks.
8. The study's eligible learning tasks are explicitly declared.
9. Train-only feature selection has been performed after split assignment.
10. The resulting split manifest is frozen and hashed.

## Planned pilot split

The three-study set is **not** sufficient by itself for the final study-held-out benchmark because a three-way split leaves very few independent studies per partition. It is therefore an integration/pipeline pilot.

Before the first scientific claim, expand the cohort with additional independent studies so that held-out-study evaluation has several independent test studies. CardiBench's existing MI/reference and study-heldout manifests are useful candidate sources, but their sample-level biological grouping must be reconciled first.

## What must not happen

- No random cell-level split for the primary benchmark.
- No regeneration label derived from a hand-written marker list and then used as a discovery target.
- No test-set-driven gene selection.
- No study ID or other provenance field used as biological input.
- No assumption that scRNA-seq and snRNA-seq are biologically identical measurements.
- No longitudinal "same-cell" claim from ordinary destructive single-cell sampling.

## Output artifacts

The audit stage writes:

```text
runs/real-data-pilot-v0.1/manifest_audit.json
```

Later materialization will add dataset/sample manifests, checksums, frozen split assignments, and feature-selection records.
