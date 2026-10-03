# Step 14 — Perturbation prediction

## Scientific purpose

Step 14 adds a predictive layer that estimates how a cardiac state changes under a specified perturbation. The target is the latent response

```text
Δz = z_perturbed − z_baseline
```

rather than a direct claim about a causal biological effect.

## Inputs

The predictor consumes:

- a baseline CardiLearn shared latent state `z_baseline`;
- perturbation identity;
- perturbation type;
- dose in dataset-native units;
- exposure duration in dataset-native units.

Continuous covariates must be normalized using training data only when the eventual training pipeline is connected to a real dataset.

## Outputs

For each latent dimension the model returns:

- `delta_mean`: predicted mean perturbation-induced latent change;
- `delta_logvar`: predicted log variance.

A predicted counterfactual state is then

```text
z'_pred = z_baseline + Δz_pred
```

with latent uncertainty derived from the predicted variance.

## Training objectives

The implementation provides:

1. latent delta MSE for absolute response accuracy;
2. cosine direction loss for response-vector direction;
3. heteroscedastic Gaussian negative log likelihood for calibrated response uncertainty.

The default configuration weights these terms without assuming any particular biological scale.

## Evaluation policy

Report at minimum:

- delta MSE;
- delta MAE;
- cosine similarity between predicted and observed response vectors;
- response-direction accuracy.

Evaluation must distinguish seen versus unseen perturbations. Where the experimental design permits it, the primary generalization test should hold out entire perturbation identities or studies rather than random cells.

## Biological guardrails

This module is explicitly predictive. A high prediction score does not establish that the predicted perturbation caused the observed transition. Causal interpretation requires an independent experimental design with appropriate controls, biological replicates, and validation.

The model also does not yet predict raw post-perturbation gene expression directly. Gene-expression counterfactuals require a separately validated decoder or response model and must not be inferred solely from latent arithmetic.

## Planned progression

After the real-data representation is locked, Step 14 should progress through:

1. paired baseline/perturbed observations;
2. train/validation/test splits at the biological-unit level;
3. seen/unseen perturbation benchmarks;
4. dose-response interpolation tests;
5. temporal response tests;
6. independent perturbation validation.

Only after those tests should CardiLearn expose perturbation-response claims beyond the latent predictive benchmark.
