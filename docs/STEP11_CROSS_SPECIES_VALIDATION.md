# Step 11 — Cross-species validation

## Goal
Measure whether CardiLearn learns conserved cardiac biology rather than species-specific signatures.

## Evaluation modes

### Leave-one-species-out
Train on two species and evaluate frozen representations on the held-out species.

### Species adversarial evaluation
Train a latent representation that predicts cardiac state while reducing recoverable species information.

### Trajectory conservation
Evaluate whether developmental and injury trajectories align across species.

## Rules
- Species labels are metadata only and are never used for benchmark leakage shortcuts.
- Test species remain unseen during representation fitting.
- Linear probes are used for frozen latent evaluation.

Initial species:
- mouse
- pig
- human

Future external challenge:
- zebrafish
