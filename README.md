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

The repository also exposes Python model/training APIs. The prototype model should be used for lightweight software testing where supported by the training scripts.

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
