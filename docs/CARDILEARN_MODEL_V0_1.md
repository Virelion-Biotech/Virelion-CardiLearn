# CardiLearn model v0.1

## Status

`main` now tracks the new CardiLearn research direction developed in September 2026. The obsolete standalone PyTorch MLP in `cardilearn/deep.py` has been removed. The new prototype lives under `cardilearn/prototype/`.

The existing scikit-learn MLP, ridge, logistic-regression, histogram-gradient-boosting, PCA, provenance, data-contract, splitting, and benchmark utilities remain because they are baselines or research infrastructure—not the CardiLearn model itself.

## Scientific target

CardiLearn is intended to learn a transferable cardiac cell-state representation from transcriptomic measurements and biological context. The initial biological axes are developmental state, maturation, injury/remodeling, and later regenerative-associated state.

The representation is not an absolute clinical or biological probability and the prototype makes no causal or clinical claims.

## v0.1 architecture

```text
expression
   -> gene tokens
   -> learned program-query cross-attention
   -> molecular representation
   + species/assay context
   -> contextual modulation
   -> shared/private latent representation
   -> maturation / injury / cell-state heads
   -> reconstruction decoder
```

Initial dimensions:

- 2,000 input genes for the CPU pilot
- 64-dimensional gene tokens
- 16 learned molecular program queries
- 128-dimensional shared latent
- 32-dimensional private latent

The program layer is program-query -> gene-key/value cross-attention, giving O(KG) rather than O(G^2) attention at prototype scale.

## Training curriculum

The first stage uses reconstruction, view invariance, and anti-collapse objectives. Biological-state supervision is then added. Regeneration ranking, cross-species alignment, and population-state transition learning are deliberately deferred until the core representation passes its benchmarks.

## Data constraints

The biological hierarchy is study -> subject -> sample -> cell. Cells are learning observations, but biological confidence and primary evaluation are based on independent biological units. Whole studies remain intact across primary train/validation/test partitions.

Gene selection and any trainable preprocessing are fit on the training partition only.

## Baselines

The initial mandatory baseline suite is:

1. PCA + linear probes
2. plain MLP
3. plain autoencoder
4. CardiLearn prototype

Additional VAE and generic attention baselines should only be added after the first comparison establishes whether the structured architecture is useful.

## Research safeguards

The prototype should be considered successful only if it improves held-out biological generalization, not merely reconstruction loss or in-sample classification. Study-ID and technical robustness probes are diagnostic safeguards against shortcut learning.

Regeneration is initially treated as a relational, sample-level learning problem. It is not encoded as a hand-written marker score or interpreted as a probability that a cell will regenerate tissue.
