# Step 15 — Benchmarking against existing methods

## Purpose

Step 15 establishes a reproducible comparison between CardiLearn and simpler, established baselines. It is a benchmark protocol, not a claim that CardiLearn is superior.

## Required comparison set

The default matrix is:

1. **PCA + linear probe** — compressed linear baseline.
2. **Plain MLP** — nonlinear supervised baseline.
3. **Plain autoencoder** — reconstruction-based representation baseline.
4. **CardiLearn** — structured representation-learning model.

The same biological task, dataset, split manifest, preprocessing boundary, model-selection rule, seed set, and evaluation metric must be used across models wherever technically appropriate.

## Leakage and model selection

The test partition is locked before experimentation. Hyperparameters and model selection use validation data only. Train-fitted preprocessing must not inspect validation/test observations. Biological grouping remains authoritative: related cells from one biological sample cannot be treated as independent train/test evidence.

## Repeated runs

The default seed set is `13, 42, 73, 101, 131`. Report mean and standard deviation across seeds. When paired seed-level scores are available, `compare_seeded_scores` provides a Wilcoxon signed-rank comparison. The resulting p-value is descriptive for this benchmark and must not be presented as independent biological evidence.

## Representation benchmark

For representation-learning evaluation, CardiLearn must be compared on downstream probes rather than reconstruction loss alone. Recommended primary probes are:

- biological-state classification;
- maturation regression/ranking;
- injury-state prediction;
- cross-species transfer;
- study-ID shortcut prediction as a negative control.

A model that reconstructs well but fails held-out biological probes has not demonstrated a useful biological representation.

## Capacity and budget fairness

Model comparisons should document parameter counts, input dimensions, optimizer, learning-rate schedule, training epochs/steps, early-stopping policy, and compute budget. Where exact architectural matching is impossible, the difference must be reported explicitly rather than hidden.

## Statistical reporting

For each model and benchmark, retain:

- per-seed test metrics;
- mean ± SD;
- confidence intervals when a sufficient number of independent evaluation units exists;
- paired model deltas where the same seed/split applies;
- number of biological subjects/samples/cells contributing to the evaluation.

Do not use repeated cells from the same biological sample as independent statistical replicates.

## Claims boundary

Allowed after a locked evaluation:

- held-out predictive performance;
- robustness across seeds;
- relative performance against declared baselines.

Not established by this benchmark alone:

- causal superiority;
- clinical utility;
- regenerative efficacy;
- biological mechanism.

Those require additional experimental or external validation.

## Current status

Step 15 implementation is **protocol-ready**, but no real-data benchmark winner is claimed. The benchmark becomes an empirical result only after the declared datasets are locked, materialized, trained, and evaluated under this protocol.
