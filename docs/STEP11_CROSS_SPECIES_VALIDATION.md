# Step 11 — Cross-species validation

## Goal
Measure whether CardiLearn learns conserved cardiac biology rather than species-specific signatures.

## Evaluation modes

### Leave-one-species-out
Train on two species and evaluate frozen representations on the held-out species. The held-out species must contribute no samples to representation fitting.

The current v0.1 prototype accepts a learned species embedding as encoder context. Therefore a strict leave-one-species-out claim cannot be made by feeding an unseen species index into that embedding table: its representation would be untrained. Until the encoder supports species-agnostic inference (or a deliberately trained unknown-species pathway), LOSO evaluation is performed on already-computed frozen representations or on an encoder configuration whose inference path does not require the held-out species identity.

### Species adversarial evaluation
Train a latent representation that predicts cardiac state while reducing recoverable species information. This is an optional extension, not a prerequisite for reporting transfer.

### Trajectory conservation
Evaluate whether developmental and injury trajectories align across species. Trajectory scores are descriptive sanity checks and do not establish causality.

## Rules
- Species labels are metadata and are not treated as biological targets.
- Test species remain unseen during representation fitting for strict LOSO.
- Linear probes are fitted only after the representation is frozen.
- Subject/sample/study-family identity remains protected by the Step 10 leakage policy.
- Cross-species validation must not silently manufacture subject identities or infer missing biological replicate mappings.
- Adversarial species removal must not be enabled by default; it can erase legitimate species biology and therefore requires a baseline transfer comparison.
- Zebrafish is reserved as a future external challenge species rather than being added to the initial human/mouse/pig training pool.

## Initial species
- mouse
- pig
- human

## Future external challenge
- zebrafish
