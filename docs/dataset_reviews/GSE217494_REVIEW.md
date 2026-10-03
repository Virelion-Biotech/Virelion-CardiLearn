# GSE217494 — biological review record

## Status

**Candidate / RNA-only external human anchor; not approved for data lock.**

## Source evidence

- GEO series: `GSE217494`
- BioProject: `PRJNA899108`
- Organism: `Homo sapiens`
- Assay: CITE-seq
- 22 explanted human left-ventricle hearts
- Groups reported by GEO: 6 healthy donors, 4 acute MI, 6 chronic ischemic cardiomyopathy, 6 non-ischemic cardiomyopathy
- GEO provides 44 samples: 22 GEX libraries paired with 22 antibody/ADT libraries.
- Processed 10x GEX matrices and a cell metadata file are available.

## CardiLearn modality decision

**Use RNA expression only for the current representation-learning model.**

The antibody/ADT component is valuable but must not be silently concatenated with RNA features. The current CardiLearn input contract is transcriptomic. If a multimodal branch is added later, ADT must receive its own explicit preprocessing, normalization, provenance, and validation pathway.

## Biological independence

This is substantially stronger than simply treating 44 GEO samples as 44 biological units: GEO explicitly reports 22 explanted hearts/donors, with paired GEX and Ab libraries. Therefore the biological unit for leakage control should be the donor/heart, not the GEO sample accession and definitely not the GEX/Ab library pair as separate observations.

The current lock pipeline still requires an explicit donor-to-sample mapping in the assembled metadata before approval.

## Candidate canonical fields

- `species = Homo sapiens`
- `study_id = GSE217494`
- `study_family_id = kuppe_human_mi`
- `assay = CITE-seq_RNA` for the RNA branch
- tissue/region = left ventricle
- condition derived from the explicit donor-level group metadata
- `subject_id = donor/heart identifier` only after explicit source reconciliation

The chronic ischemic, acute MI, non-ischemic cardiomyopathy, and healthy groups must remain distinct labels at ingestion. Do not collapse all disease groups into a generic `injury` class without a task-specific mapping.

## Critical confounding considerations

The human dataset is an external clinical anchor rather than a clean controlled regeneration experiment. Cardiomyopathy, disease chronicity, donor characteristics, tissue state, and clinical heterogeneity may be substantial sources of variation. Consequently it is especially appropriate for external/generalization evaluation and human-state anchoring, but biological conclusions about regeneration must not be inferred solely from this dataset.

## Lock decision

**NOT ELIGIBLE FOR LOCK YET.**

Required:

1. reconcile donor IDs and all paired GEX/ADT libraries;
2. confirm the RNA matrix and metadata join exactly;
3. retain the clinical disease categories without over-collapsing labels;
4. perform cell/sample QC;
5. determine whether this study is used in training or reserved as an external human evaluation cohort;
6. prevent donor-level leakage across all partitions.

## Acquisition policy

The processed GEX MTX/TSV files and `GSE217494_cell.metadata.csv.gz` are sufficient for the initial RNA-only CardiLearn pipeline. Do not download the multi-terabyte SRA archive merely for this stage.
