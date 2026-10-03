# CardiLearn real-data manifests

These files describe source datasets supplied to CardiLearn without redistributing the original GEO archives.

Each manifest records:

- source accession and modality
- exact local raw-archive SHA-256
- sample-level labels and grouping keys
- matrix dimensions and sparsity
- sample-level QC summaries calculated from the supplied 10x Matrix Market files

The current verified local inputs are:

- `GSE153480_RAW.tar` — eight neonatal mouse cardiac scRNA-seq samples, MI or Sham across P1/P8 and postoperative day 1/3.
- `GSE216211_RAW.tar` — three adult mouse cardiac immune-cell scRNA-seq samples: Sham, MI, and MI-E.

The raw archives are intentionally **not committed to GitHub**. They should be downloaded or supplied locally and SHA-256 verified against these manifests before preprocessing.

## Modeling boundary

For cell-level ML, `group_id` is the biological sample accession. All cells from one sample must remain in exactly one split. This prevents cell-level leakage from turning a sample-level treatment label into an artificially easy task.

For cross-study generalization, study/accession is an additional held-out boundary.

## Important QC observation

GSE153480 has heterogeneous QC. In particular, GSM4644950 (P1 Sham) and GSM4644954 (P8 Sham) have substantially higher upper-tail mitochondrial fractions than most other samples. These samples are flagged for explicit QC sensitivity analysis rather than silently removing them.
