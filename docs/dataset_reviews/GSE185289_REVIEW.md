# GSE185289 — biological review record

## Status

**Candidate / not approved for data lock.**

## Source evidence

- GEO series: `GSE185289`
- Organism: `Sus scrofa`
- Assay: single-nucleus RNA-seq
- Study: Nakada et al., myocardial regeneration after apical resection / MI
- GEO reports 56 samples and multiple experimental groups spanning fetal and postnatal maturation, control hearts, apical resection, MI, and combined injury.
- The published analysis reports 218,945 high-quality nuclei overall and 94,844 cardiomyocytes with complete gene-nucleus data.

## Experimental groups reported by GEO

- FH
- CTL-P1
- CTL-P28
- CTL-P56
- ARP1-P28
- ARP1-P56
- MIP28-P56
- ARP1-MIP28-P30
- ARP1-MIP28-P35
- ARP1-MIP28-P42
- ARP1-MIP28-P56

`ARP1` denotes apical resection at P1; `MIP28` denotes LAD ligation at P28. The combined-injury model is therefore biologically distinct from ordinary MI and must not be collapsed into a generic regeneration label.

## Eligibility assessment

**Strong candidate for the cross-species regeneration/maturation arm**, particularly for cardiomyocyte state representation. However, it should not yet enter a locked split.

### Major unresolved issue: biological independence

The 56 GEO samples do not, by themselves, establish the independent-animal hierarchy required by CardiLearn. Sample-level identifiers must be reconciled to animal/heart identifiers using the original study metadata/source tables.

Therefore:

- do not equate GSM accession with `subject_id` automatically;
- do not count 56 samples as 56 biological replicates;
- retain `subject_confidence = unverified` until explicitly reconciled.

### Assay boundary

This dataset is snRNA-seq and should retain `assay = snRNA-seq`. It must not be silently treated as scRNA-seq merely because both produce gene-by-observation matrices.

### Biological labels

Candidate fields after review:

- `species = Sus scrofa`
- `assay = snRNA-seq`
- `study_id = GSE185289`
- `study_family_id = nakada_pig_regeneration`
- maturation derived from explicit fetal/postnatal age metadata
- injury states derived from explicit experimental group definitions

Do not encode ARP1-MIP28 as simply `regeneration=true`; preserve the actual perturbation history because it represents a sequential injury model.

## Lock decision

**NOT ELIGIBLE FOR LOCK YET.**

Required:

1. reconcile each sample to its biological heart/animal;
2. verify group and harvest-age metadata;
3. identify technical versus biological replicates;
4. identify the processed expression file actually selected for CardiLearn;
5. verify cell/nucleus annotation provenance;
6. run matrix-level and sample-level QC;
7. establish the study-family relationship to any other Nakada/Nguyen pig datasets before splitting.

## Acquisition policy

Prefer the processed GEO expression files over raw sequencing for the initial representation-learning cohort. Raw SRA data are not required for the current objective unless a later reprocessing analysis specifically warrants them.
