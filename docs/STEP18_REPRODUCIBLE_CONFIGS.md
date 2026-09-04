# Step 18 — Reproducible Configurations

## Purpose

A CardiLearn scientific run must be reconstructable from explicit inputs rather than hidden Python defaults or notebook state. Step 18 therefore treats the experiment configuration, dataset identity, frozen split identity, seed set, and code revision as first-class provenance.

## Required provenance

Every locked experiment should record:

1. versioned configuration schema;
2. dataset source and SHA256 fingerprint;
3. biological unit used for splitting;
4. frozen split manifest and its fingerprint;
5. training/validation/test semantics;
6. preprocessing fit policy;
7. complete seed set and primary seed;
8. Git commit;
9. Python/runtime environment;
10. output artifact locations.

The template is `configs/reproducibility_v1.yaml`.

## Configuration rules

- Configuration is declarative and versioned.
- Preprocessing is fitted on training data only.
- Model selection uses validation data only.
- The test set is locked until final reporting.
- Repeated biological units are split by biological group rather than by individual cell when appropriate.
- Dataset and split fingerprints are retained with results.
- Manual preprocessing not represented in configuration is considered an unreproducible analysis step.

## Fingerprints

`cardilearn.reproducibility` provides deterministic SHA256 fingerprints for mappings, ordered ID manifests, and pandas dataframes. Configuration fingerprints are canonicalized before hashing so dictionary key order does not change identity.

A reproducibility manifest can be created with `make_manifest(...)`, saved with `save_manifest(...)`, and validated on reload with `load_manifest(...)`.

## Scientific interpretation

A reproducibility manifest establishes **what was run and what data/configuration were used**. It does not establish biological validity, causal inference, clinical utility, or model superiority. Those require the locked validation and experimental evidence described in the benchmark, interpretability, and perturbation protocols.

## Example

```python
from cardilearn import make_manifest, save_manifest

manifest = make_manifest(
    config={"model": "cardilearn", "task": "classification"},
    config_name="cardilearn_real_data_v1",
    data_fingerprint="...64-char-sha256...",
    split_fingerprint="...64-char-sha256...",
    seeds=[13, 42, 73, 101, 131],
    primary_seed=42,
    git_commit="<locked-commit>",
)
save_manifest(manifest, "artifacts/reproducibility/experiment_manifest.json")
```

The example intentionally leaves fingerprints and commit identity to the real experiment rather than inventing them.
