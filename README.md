# Virelion-CardiLearn

CardiLearn is a Python research codebase for cardiac transcriptomic representation learning and downstream prediction. It contains a prototype implementation and a larger transcriptome-scale research architecture.

## What it contains

- Transcriptome-scale gene inputs.
- Learned molecular programs.
- Program-level Transformer processing.
- Shared and private latent representations.
- Molecular reconstruction and biological-state prediction.
- Interpretability and perturbation-prediction modules.
- Leakage-aware biological splitting.
- Benchmark and provenance contracts.

The larger architecture is a research target, not a validated cardiac foundation model.

## Installation

```bash
pip install -e '.[dev,torch]'
```

For development/testing, the command above installs the development and PyTorch extras required by the full test suite. The `torch` extra is optional for workflows that do not import the torch-backed model modules.

## Usage

```bash
cardilearn models --task classification
cardilearn validate --data data.csv --target label
cardilearn train --data data.csv --target label --output runs/example
cardilearn benchmark-info --definition configs/benchmark_v1.yaml
```

For representation-learning evaluation, use `configs/representation_benchmark_v1.yaml`. The research model is evaluated from frozen embeddings rather than by reconstruction loss alone, and repeated cells/nuclei can be aggregated to biological sample before the primary probe.

The repository also exposes Python model/training APIs. The prototype model should be used for lightweight software testing where supported by the training scripts.


### Colab real-data validation

Use the `notebooks/CardiLearn_v0_5_RealData_Validation_Colab.ipynb` notebook for the locked T4 workflow. It covers sparse single-cell/single-nucleus loading, hierarchy-safe study-family splitting, train-only gene selection, CardiLearnResearch training, frozen sample-level embeddings, PCA/autoencoder baselines, bootstrap/permutation statistics, and provenance export.

The notebook requires a user-supplied locked dataset bundle; it does not invent or silently resolve ambiguous biological metadata. Large raw datasets and model checkpoints remain local to the Colab run unless the optional lightweight results push is enabled.

### Source-rescue audit

Before spending cloud compute on SRA reprocessing, run `scripts/source_rescue_audit.py` or `notebooks/Step3_Source_Rescue_Audit_Colab.ipynb`. It checks the locked GSMs against GEO supplementary candidates, NCBI/ENA SRA mappings, EMBL-EBI ArrayExpress/BioStudies and Expression Atlas evidence, optional ARCHS4 H5 sample availability, and recount3 project/run availability. It is fail-closed: ARCHS4 Kallisto-derived rounded values are not treated as strict raw counts, and recount3-derived counts are reported as derived rather than original submitter counts. Candidate sources still require content-scale, exact sample-mapping, provenance, and SHA-256 validation before entering Step 3 external sources.

### SRA raw-count rescue

The four locked cohorts that fail the strict raw integer-count contract are now covered by a reproducible raw-read rescue workflow:

`GSE232259`, `GSE52313`, `GSE186875`, and `GSE308783`.

Run `scripts/step3_sra_reprocess.py` with `--step resolve` to create the exact GEO GSM → SRX → ENA SRR → FASTQ manifest, then `--step all` on a sufficiently provisioned Linux machine or Colab high-RAM runtime. The workflow uses pinned Mus musculus GRCm39 / Ensembl 112, fastp QC with read-preserving defaults, STAR, and featureCounts. It records source checksums, reference fingerprints, per-sample completion markers, and a validated `external_count_sources.json` that the existing Step 3 baseline runner can consume.

See `docs/SRA_REPROCESSING_V1.md`, `configs/sra_reprocessing_v1.json`, `envs/sra_reprocessing_v1.yml`, and `notebooks/Step3_SRA_Reprocess_Colab.ipynb`.

## Inputs and outputs

**Inputs:** transcriptomic gene/value matrices, sample metadata and biological grouping labels, task targets, model/training configuration, and benchmark definitions.

**Outputs:** learned representations, predictions, reconstructions, training/evaluation artifacts, model checkpoints, and provenance/benchmark records.

The intended biological hierarchy is study family → study → subject/donor/animal → sample/library → cell/nucleus. Cells are observations and are not automatically independent biological replicates.

## Validation

Software tests establish implementation behavior, not biological validity. Scientific evaluation requires locked datasets, biologically appropriate held-out splits, baseline comparisons, cross-study or cross-species testing where relevant, and independent biological validation for biological claims.

The proposed comparison includes PCA + linear probe, MLP, autoencoder, and the larger model; the larger model should be retained only if it improves held-out biological generalization.

## Limitations

The large architecture is not a validated cardiac foundation model. Model capacity, parameter count, reconstruction quality, or training loss do not establish biological significance or causal validity. Perturbation predictions are hypotheses until independently tested. Results can be affected by study-family leakage, batch effects, metadata errors, and distribution shift.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.
