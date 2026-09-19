# SRA raw-count rescue pipeline (v1)

This pipeline reprocesses the four locked benchmark cohorts whose GEO processed expression sources do not satisfy the strict raw integer-count contract:

- `GSE232259` — 12 samples
- `GSE52313` — 8 samples
- `GSE186875` — 6 samples
- `GSE308783` — 7 samples

## Scientific contract

The input lineage is fixed:

`GEO GSM -> GEO SRA relation -> SRX -> ENA SRR -> raw FASTQ -> fastp/QC -> STAR -> featureCounts -> matrix`

The primary retrieval path uses ENA-generated FASTQ for the exact SRR records returned by ENA. SRA Toolkit is a fallback when ENA does not expose generated FASTQ URLs.

The pipeline records ENA-provided FASTQ MD5 and byte-size metadata and verifies them after download when available. Completed artifacts are reused only when their completion marker and expected file are present.

The count model is:

- reference: Mus musculus GRCm39
- annotation: Ensembl release 112
- feature type: `exon`
- grouping attribute: `gene_id`
- strand: unstranded (`-s 0`)
- paired-end: inferred per sample from ENA metadata
- paired-end counting: `-p --countReadPairs`
- multimappers: excluded by featureCounts defaults
- fractional assignment: disabled

The default QC stage writes fastp reports while disabling trimming/filtering so reads are not silently altered before alignment.

## Tool versions

- SRA Toolkit `3.4.1`
- fastp `1.3.6`
- STAR `2.7.11b`
- Subread/featureCounts `2.1.1`

These are intentionally pinned.

## Resource expectation

Do not run the complete four-cohort alignment on an 8 GB laptop.

Use Linux x86-64 with about 64 GiB RAM and roughly 500 GiB working storage as a practical starting point. STAR is the dominant resource consumer; the pipeline does not use the T4 GPU.

Run one accession at a time when storage is constrained:

``bash
python scripts/step3_sra_reprocess.py \
  --step all \
  --accessions GSE232259 \
  --work-dir /mnt/cardilearn_step3 \
  --threads 16
``

Repeat for `GSE52313`, `GSE186875`, and `GSE308783`.

## Outputs

``text
step3_reprocessed/
  metadata/
    locked_run_manifest.tsv
    locked_run_manifest.json
    locked_run_manifest.sha256
  raw/
    <GSE>/<GSM>/<SRR>/*.fastq.gz
  samples/
    <GSE>/<GSM>/*.fastq.gz
  qc/
    <GSE>/<GSM>.fastp.html
    <GSE>/<GSM>.fastp.json
  reference/
    *.fa
    *.gtf
    STAR_index/
    reference_fingerprint.json
  alignments/
    <GSE>/<GSM>/*.bam
  counts/
    <GSE>/<GSM>.featureCounts.txt
  matrices/
    <GSE>.raw_counts.tsv.gz
  validation/
    <GSE>.json
  external_count_sources.json
``

The generated `external_count_sources.json` is compatible with the existing Step 3 runner through `STEP3_EXTERNAL_SOURCES`.

## Stage meanings

`resolve` fetches GEO metadata and exact GSM -> SRX relationships, then resolves SRX -> SRR through ENA.

`download` fetches exact FASTQ files with retry/resume and checksum/size verification. It falls back to `prefetch + fasterq-dump` only when needed.

`merge` combines multiple raw FASTQ files belonging to one locked biological sample without re-encoding the gzip members.

`qc` generates fastp JSON/HTML reports. v1 leaves reads unchanged by default.

`align` builds the locked GRCm39/Ensembl-112 STAR index once per workspace and aligns each sample.

`count` runs featureCounts independently per sample so mixed single/paired layouts cannot accidentally share one global paired-end flag.

`validate` requires exact locked GSM columns, Ensembl mouse gene IDs, nonnegative integer counts, and complete sample coverage. Only after validation is the accession entered into `external_count_sources.json`.

## Fail-closed rules

The pipeline does not infer biological identity from filename order, sample-number order, or SRR ordering.

It does not round FPKM/TPM/CPM/normalized matrices.

It does not promote ARCHS4 or recount3 to original submitter raw-count status.

It does not alter the frozen benchmark split, variable-gene selection rule, or test-set policy.

## Documentation

- NCBI SRA Toolkit: https://github.com/ncbi/sra-tools
- ENA Portal file reports: https://ena-docs.readthedocs.io/en/latest/retrieval/programmatic-access/file-reports.html
- fastp: https://github.com/OpenGene/fastp
- STAR: https://github.com/alexdobin/STAR
- Subread/featureCounts: https://subread.sourceforge.net/
- Ensembl release 112: https://www.ensembl.org/info/news/