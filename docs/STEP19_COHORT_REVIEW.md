# Step 19 — Cohort Review and Lock Decision Framework

## Purpose

This document defines the evidence required before a public cardiac single-cell/single-nucleus dataset can enter CardiLearn training or a locked external evaluation.

A GEO/SRA sample count is **not** a biological replicate count. The canonical hierarchy is:

`study_family_id → study_id → subject_id → sample_id → cell_or_nucleus_id`

No split may occur below `study_id`, and related studies must be grouped by `study_family_id` when biological independence has not been demonstrated.

## Current decision state

The registry remains `candidate_sources_not_locked`.

### GSE153480 — Wang/Olson neonatal mouse regeneration

**Use:** high-priority regenerative mouse anchor.

NCBI reports eight scRNA-seq samples: P1/P8, MI/sham, 1/3 days after surgery. The associated BioProject reports eight BioSamples and eight SRA experiments. This establishes dataset structure, but does not by itself establish eight independent animals for every analytical purpose. The study's scRNA-seq series must remain separate from the related scATAC series.

**Training status:** candidate pending subject-level verification.

**Required evidence before lock:**
- exact animal/biological replicate mapping for all eight samples;
- confirmation that no animal contributes to multiple nominally independent samples in a way that crosses the planned split;
- source-verified P1/P8, MI/sham, and timepoint labels;
- cell-level QC thresholds selected without using held-out biology;
- final sample and study-family assignment reviewed.

### GSE185289 — pig regeneration/maturation

**Use:** high-value mammalian cross-species anchor.

NCBI reports 56 GEO samples spanning fetal, control, apical resection, MI, and combined injury groups. Sample names encode animal-like identifiers in many records, but encoded names must still be reconciled against the source metadata rather than treated as proof of independence. The dataset contains multiple platforms and a large processed Seurat object plus raw/processed supplementary data.

**Training status:** candidate pending subject-level reconciliation.

**Required evidence before lock:**
- verified animal ID for every sample;
- mapping of animal → experimental group → harvest day → library/sample;
- identification of technical/repeated libraries;
- platform/assay reconciliation;
- explicit cardiomyocyte-only versus whole-heart scope decision;
- one-to-one orthology policy for pig-to-canonical genes;
- confirmation that no animal crosses train/validation/test.

### GSE217494 — human cardiac disease

**Use:** human external anchor; potentially a training component only after donor-level reconciliation.

CITE-seq protein measurements must remain separate from the RNA representation unless a dedicated multimodal model is introduced.

**Training status:** candidate / external-anchor preferred.

**Required evidence before lock:** donor-level mapping, tissue/region mapping, assay branch definition, disease-state normalization, donor independence, and RNA feature compatibility.

## Dataset-family rules

1. Do not call cells independent observations for split purposes.
2. Do not call libraries independent biological replicates without source evidence.
3. Do not infer subject IDs from filenames when authoritative metadata exist.
4. Do not merge scRNA-seq, snRNA-seq, CITE protein, spatial transcriptomics, ATAC-seq, and bulk RNA into one expression matrix without an explicit modality model.
5. Developmental datasets may provide maturation supervision without an injury label.
6. Reprogramming and conduction-system datasets must not be relabeled as generic spontaneous regeneration.
7. Cross-species training requires an explicit, frozen gene-ID and orthology mapping before expression matrices are combined.
8. Variable-gene selection, normalization, dimensionality reduction, imputation, and any other learned preprocessing must be fitted on training observations only.
9. A dataset that fails any lock criterion remains a candidate or external-only source.
10. No empirical biological claim may be made from software/CI success.

## Required lock evidence package

A candidate becomes lock-eligible only when the following artifacts exist outside the repository's raw-data area:

- source acquisition manifest;
- SHA256 for every processed input;
- canonical sample manifest;
- verified subject/sample hierarchy;
- study-family registry;
- gene-ID mapping and orthology manifest;
- cell-level QC report;
- sample-level QC report;
- label audit report;
- reviewed train/validation/test split manifest;
- deterministic split fingerprint;
- expression and metadata fingerprints;
- final lock manifest.

The repository must store the code, schemas, templates, and fingerprints—not raw sequencing archives or large expression matrices.

## Current next action

Acquire and audit the explicitly declared processed GSE153480 10x files first, reconcile their biological hierarchy, run structural/sample QC, and produce a **candidate-only** cohort report. Do not approve the split automatically.
