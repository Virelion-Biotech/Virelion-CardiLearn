# Step 19 — Training Readiness

## Do not train yet

The repository currently contains candidate public studies and a metadata-first
acquisition/reconciliation pipeline. A study accession is **not** a data lock.
Training becomes authorized only after all of the following are true:

1. Source metadata and expression matrices have been materialized outside Git.
2. Every observation has a stable `study_id → subject_id → sample_id → cell_or_nucleus_id` lineage.
3. Condition, injury, maturation, and cell type labels are explicitly reconciled or the observation is excluded.
4. Technical replicates are not treated as independent biological replicates.
5. Genes are mapped with the frozen one-to-one policy; ambiguous mappings are excluded.
6. The expression matrix and metadata have matching observation order and stable fingerprints.
7. Study-level splits are frozen before train-only feature selection.
8. No subject/sample/study-family identity crosses the locked partitions.
9. The locked test partition is untouched during model fitting and model selection.
10. The resulting data-lock manifest is reviewed and marked locked.

Until those gates pass, a successful optimizer run would only demonstrate that the
software can train—not that CardiLearn learned a biologically valid representation.

## Recommended first training scale

The first real run should be a **pilot**, not the final multi-species training run:

- 2,000 genes selected using training observations only.
- `float32` expression.
- batch size 32 initially; increase only after checking GPU memory.
- 10 epochs for the smoke/pilot run.
- primary seed 42, followed by the reproducibility seed set only after the pipeline is stable.
- validation/test remain completely outside the optimizer.

The new sparse ingestion primitives keep large 10x matrices sparse during loading and
train-only variable-gene selection. This is important because densifying a full
single-cell matrix with tens of thousands of genes can exceed a 13 GB system-RAM
runtime even when the eventual 2,000-gene training matrix is small enough.

## Colab / GPU

A 13 GB **system RAM** Colab runtime can be sufficient for the first train-ready
2,000-gene pilot if the curated matrix is kept to a manageable observation count and
we avoid dense copies of the full gene matrix. GPU memory is separate from system
RAM. Colab does not guarantee a particular GPU, GPU memory, runtime duration, or
availability; these vary with the current product tier and resource availability.

When training is authorized, use a GPU runtime and verify the assigned hardware
inside the notebook before starting. If a local GPU is available, Colab can also
connect to a local runtime, allowing the notebook to use the user's own hardware.

## When to switch from pilot to serious training

Use the pilot to establish:

- the data-load memory profile;
- training throughput;
- loss stability;
- checkpoint/restart behavior;
- whether 2,000 genes and the current architecture fit comfortably on the GPU.

Only after the locked pilot passes the Step 15 benchmark and cross-study/species
checks should we launch the larger multi-seed training campaign.
