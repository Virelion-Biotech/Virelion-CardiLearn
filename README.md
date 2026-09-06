# Virelion-CardiLearn

CardiLearn is a Python research codebase for cardiac transcriptomic representation learning and downstream prediction. The repository contains a small prototype and a larger transcriptome-scale research architecture.

## Scope

The research architecture is designed around:

- transcriptome-scale gene inputs;
- learned molecular programs;
- program-level Transformer processing;
- shared and private latent representations;
- molecular reconstruction and biological-state prediction;
- interpretability and perturbation-prediction modules;
- leakage-aware biological splitting;
- benchmark and provenance contracts.

The larger architecture is a research target, not a validated cardiac foundation model.

## Model architecture

```text
transcriptome
    ↓
gene/value embeddings
    ↓
learned molecular programs
    ↓
program-level Transformer
    ↓
shared + private state
    ├── maturation
    ├── injury/disease state
    ├── cell identity
    └── downstream prediction
```

The repository currently targets approximately 20,000 genes, 32 molecular programs, six Transformer blocks, a 512-dimensional shared state, and a 128-dimensional private state for the large architecture. These are configuration targets, not evidence of model performance.

## Data and leakage policy

The biological hierarchy is:

```text
study family → study → subject/donor/animal → sample/library → cell/nucleus
```

Cells are observations and are not automatically independent biological replicates. Train/validation/test boundaries must be defined at the appropriate biological grouping level. Train-derived preprocessing must not inspect held-out observations.

Candidate public studies are not a locked training or benchmark corpus until study relationships, subject/sample mapping, assay boundaries, and conditions are reconciled.

## Baselines

The initial comparison should include:

1. PCA + linear probe;
2. MLP;
3. autoencoder;
4. CardiLearnLarge.

The larger model should be retained only if it improves held-out biological generalization relative to appropriate baselines.

## Installation

```bash
pip install -e .
```

## CLI

```bash
cardilearn models --task classification
cardilearn validate --data data.csv --target label
cardilearn train --data data.csv --target label --output runs/example
cardilearn benchmark-info --definition configs/benchmark_v1.yaml
```

For lightweight software testing, the prototype model can be selected where supported by the training scripts.

## Validation

A CI pass demonstrates software correctness, not biological validity. Scientific evaluation requires locked data, held-out studies, cross-study/cross-species testing where relevant, baseline comparisons, and independent biological validation for biological claims.

## Integration

- **CardiAtlas:** study, sample, phenotype, and provenance context.
- **CardiBench:** benchmark definitions and leakage-aware splits.
- **CardiEval:** independent evaluation of model outputs.
- **CardiSim:** synthetic trajectories and mechanistic research scenarios.
- **CardiTrace:** training/run provenance.
- **CardiBridge:** cross-component contracts.
- **HeartTwin:** multimodal orchestration.

## Limitations

Model capacity, parameter count, reconstruction quality, or training loss do not establish biological significance or causal validity. Perturbation predictions are hypotheses until independently tested.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.

## Citation

Cite the repository release and all training/benchmark datasets and source publications used.
