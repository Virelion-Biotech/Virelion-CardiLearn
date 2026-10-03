# CardiLearn Architecture

CardiLearn is organized around five operational layers plus the research-model extensions.

1. **Ingest** — loaders and modality containers convert raw files into explicit data contracts.
2. **Validate** — dataset checks detect missing targets, duplicate IDs, missingness, and grouping problems before training.
3. **Learn** — preprocessing and estimators live inside pipelines so transformations are fit on training data only.
4. **Evaluate** — validation is used for model selection; the final test set is reserved for frozen-model assessment.
5. **Export** — run manifests, dataset cards, fingerprints, and prediction exports provide a reproducible handoff to CardiEval and other components.

## Research-model layers

```text
cardilearn/
│
├── infrastructure
│   ├── data.py
│   ├── schema.py
│   ├── splitting.py
│   ├── validation.py
│   ├── provenance.py
│   └── registry.py
│
├── representation
│   ├── research_model.py
│   ├── objectives.py
│   ├── conserved.py
│   ├── backbones.py
│   ├── singlecell.py
│   └── prototype/
│       ├── model.py
│       └── losses.py
│
├── interpretation
│   └── interpretability/
│       ├── attribution.py
│       ├── latent_analysis.py
│       ├── enrichment.py
│       ├── counterfactual.py
│       └── ortholog_conservation.py
│
├── perturbation
│   └── perturbation/
│       ├── contracts.py
│       ├── model.py
│       ├── losses.py
│       ├── response.py
│       └── evaluation.py
│
└── benchmarking
    ├── benchmark_protocol.py
    └── benchmark_suite.py
```

## Representation model

The current modular research path follows:

```text
counts
   → gene identity + value encoding
   → optional conserved-gene identity
   → memory-bounded gene→program routing
   → program Transformer / optional Mamba
   → shared + private latent state
   → factorized negative-binomial reconstruction
   + masked-gene prediction
   + maturation / injury / cell-type heads
   + optional species-adversarial head
```

The shared latent is intended to capture reusable biological state; the private latent retains context that should not be forced into the shared representation.

## Training objectives

The research direction combines reconstruction, technical-view invariance, anti-collapse regularization, biological supervision, maturation ordering/regression, and injury-state prediction. Cross-species alignment and regeneration/transition objectives are later additions that require their own locked evaluation protocols.

## Interpretability

Interpretability is downstream of a frozen model and declared analysis split:

```text
trained model
    ↓
latent / prediction target
    ├── Integrated Gradients
    ├── masking sensitivity
    ├── permutation sensitivity
    ├── latent/program analysis
    ├── gene-set enrichment
    ├── counterfactual masking
    └── ortholog-group conservation
```

Attention-derived program weights are supplementary evidence, not causal explanations.

## Perturbation prediction

The perturbation module predicts latent responses:

```text
baseline z + perturbation identity/type + dose + duration
                         ↓
                 response predictor
                         ↓
                    Δz + variance
                         ↓
                   z + Δz
```

A predicted response is not a causal effect without independent experimental validation.

## Benchmarking

The benchmark layer compares declared baselines and CardiLearn under a common protocol. The current baseline matrix is:

- PCA + linear probe;
- plain MLP;
- plain autoencoder;
- CardiLearn.

Dataset fingerprints, protocol versions, seeds, and metrics are recorded. Validation can guide model selection; the final test set remains locked.

## Data hierarchy and leakage

```text
study
  ↓
subject / donor / animal
  ↓
sample / library
  ↓
cell / nucleus
```

A biological unit must not cross the declared evaluation boundary. Cell count is not biological replicate count. Train-derived preprocessing must never inspect held-out data.

## Dependency boundaries

The minimal package does not require PyTorch. Torch-backed research components are lazy-loaded, allowing metadata, validation, splitting, provenance, and benchmark infrastructure to run in lightweight environments. Optional dependencies are separated in `pyproject.toml`.

## Ecosystem boundary

CardiLearn owns learning and model development. CardiBench owns canonical datasets/splits. CardiEval independently evaluates exported predictions. CardiTrace records provenance across the ecosystem. This separation is intentional: the model should not be the sole judge of its own scientific validity.


## Independent evaluation and scale

The research representation is not evaluated by its own reconstruction loss alone. Frozen embeddings are passed to declared downstream probes under the locked biological grouping protocol, with optional aggregation from cell/nucleus to sample before inference. Cross-study and cross-species transfer are evaluated separately. External transcriptomic encoders are accessed through caller-supplied adapters so repository-specific APIs are not assumed.

The high-capacity architecture remains an empirical research implementation. Parameter count, training loss, and software tests are implementation evidence, not biological validation.
