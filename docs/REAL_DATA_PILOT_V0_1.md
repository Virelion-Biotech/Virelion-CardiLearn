# CardiLearn v0.1 — Real-data pilot

## Purpose

This is the first real-data integration layer for the new CardiLearn research model. It is deliberately **candidate-only**: an accession being listed here does not mean that its expression matrix has been downloaded, its biological subject mapping has been reconciled, or that it is safe to use as a locked benchmark.

The code/data boundary is:

```text
CardiAtlas / CardiBench
        ↓
canonical study + sample metadata
        ↓
GEO metadata profile (descriptive only)
        ↓
CardiLearn real-data audit
        ↓
study-family / subject reconciliation
        ↓
feature/task eligibility
        ↓
locked split
        ↓
training matrix
```

Source expression data are not committed to GitHub. GEO acquisition records SHA-256 provenance.

## Pilot cohort

### Development/regeneration anchors

**GSE185289 — pig** (`Sus scrofa`, snRNA-seq): fetal and postnatal control states plus apical resection, MI, and combined injury groups across postnatal timepoints. This is the primary pilot source for maturation/regeneration relational supervision.

**GSE130699 — mouse** (`Mus musculus`, snRNA-seq): neonatal cardiomyocyte nuclei after MI or sham at P1/P8 with 1- and 3-day collection. It is a compact regenerative-window counterpart to the pig study.

**GSE153480 — mouse** (`Mus musculus`, scRNA-seq): neonatal MI/sham data using the same P1/P8 and 1/3-day experimental design family. It is valuable for scRNA-versus-snRNA robustness, but it is **not automatically an independent study** for held-out evaluation because it is closely linked to GSE130699 and is explicitly grouped with it in the study-family registry pending sample-overlap reconciliation.

### Human/external injury context

**GSE217494 — human** (`Homo sapiens`, CITE-seq/GEX): 22 explanted human hearts spanning healthy donors, acute MI, chronic ischemic cardiomyopathy, and non-ischemic cardiomyopathy. It is an injury/context transfer cohort, not maturation or regeneration ground truth.

### Additional injury/context candidates

**GSE135310 — mouse** (`Mus musculus`, scRNA-seq/CITE-seq): time-series cardiac leukocyte/myeloid profiles after permanent MI or sham. Useful for testing injury representations outside cardiomyocyte-focused data.

**GSE106472 — mouse** (`Mus musculus`, scRNA-seq): WT MI, IRF3-knockout MI, and WT non-MI reference. It is retained as a mechanistic external test, not a clean primary MI classifier, because genotype and condition are partly confounded.

**GSE216211 — mouse** (`Mus musculus`, scRNA-seq): cardiac macrophage profiles containing sham, MI, and MI-E. It is a cell-context robustness candidate, not a whole-heart maturation/regeneration ground truth.

**GSE269054 — human/mouse** (`Homo sapiens` and `Mus musculus`, snRNA-seq/spatial): cross-species injury study with permanent MI, I/R, and traumatic injury contexts. It is an external spatial/cell-state validation source; I/R and traumatic injury remain separate from the primary permanent-MI label.

## Why the cohort was expanded

The original three-study pilot was useful for plumbing but was too confounded: pig largely represented development/regeneration, mouse neonatal injury, and human adult injury, while modality was also partly coupled to species. The expanded candidate set now contains multiple accessions and both scRNA-seq and snRNA-seq measurements, allowing the next split-design stage to test whether learned representations survive study and assay changes.

However, **accession count is not independent-study count**. Related series can belong to the same biological study family. CardiLearn therefore maintains a versioned study-family registry and uses family identity—not raw accession count—as the conservative independence unit for external evaluation.

The cohort is still **not locked**. Several studies have incomplete public subject-level replicate mappings, and some are cell-context- or genotype-specific.

## Study-family protection

The current registry explicitly groups:

```text
wang_olson_neonatal_regeneration
├── GSE130699
└── GSE153480
```

These series both describe the Wang/Cui/Tan/Olson neonatal P1/P8 MI-versus-sham program. The registry intentionally keeps them in the same future partition until exact sample overlap and biological replicate relationships are reconciled.

Unmapped accessions receive deterministic per-accession families rather than being assumed related or independent. New linkages must be added through reviewed registry changes.

## Metadata-first rule

Before any expression matrix is used, run:

```bash
python scripts/materialize_real_data_pilot.py
python scripts/profile_geo_metadata.py
python scripts/audit_real_pilot_samples.py
```

`profile_geo_metadata.py` is intentionally descriptive. It reports raw characteristic keys/values, sample IDs, titles, and organisms without converting a filename, title, or experimental label into a biological subject ID. Reconciliation remains a separate reviewed step.

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
study family → study → subject → sample → cell
```

Technical replicates remain attached to their biological parent. Samples from the same animal/donor must never cross train/validation/test partitions. Linked study series must also remain in the same partition.

## Lock criteria

A dataset may enter a locked CardiLearn benchmark only after all of the following are true:

1. GEO/ENA accession identity is verified.
2. Sample metadata are materialized and checksum-recorded.
3. Subject/donor/animal identifiers are recovered or an explicit weaker split policy is approved.
4. Technical replicate grouping is resolved.
5. Condition and timepoint labels are normalized without ambiguous guessing.
6. Modality and tissue are verified.
7. Study-family and hierarchy leakage checks pass.
8. The study's eligible learning tasks are explicitly declared.
9. Train-only feature selection has been performed after split assignment.
10. The resulting split manifest is frozen and hashed.

## Split policy

The eight-accession candidate set is now large enough to begin deterministic split-assignment work, but the split is still **candidate-only** until subject/replicate reconciliation and task-balance checks are complete. The primary scientific split remains study-held-out; study families are the independence unit and subject-level grouping protects biological replicates inside each study.

Before the first generalization claim, reserve multiple independent study families for final testing. Do not turn the candidate list into a locked benchmark merely because it contains eight accessions.

## What must not happen

- No random cell-level split for the primary benchmark.
- No regeneration label derived from a hand-written marker list and then used as a discovery target.
- No test-set-driven gene selection.
- No study ID or other provenance field used as biological input.
- No assumption that scRNA-seq and snRNA-seq are biologically identical measurements.
- No longitudinal "same-cell" claim from ordinary destructive single-cell sampling.
- No silent subject inference from sample-name patterns.
- No mixing permanent MI, I/R, genotype perturbation, or traumatic injury into a single undifferentiated injury label.
- No counting linked accessions as independent held-out studies.

## Output artifacts

The audit stage is expected to write:

```text
runs/real-data-pilot-v0.1/manifest_audit.json
runs/real-data-pilot-v0.1/metadata_profile.json
runs/real-data-pilot-v0.1/sample_audit.json
```

Later materialization will add dataset/sample manifests, checksums, frozen split assignments, and feature-selection records.
