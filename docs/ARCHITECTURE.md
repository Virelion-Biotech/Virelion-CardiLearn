# CardiLearn architecture

CardiLearn is intentionally split into five layers.

1. **Ingest** — loaders and modality containers convert raw files into explicit data contracts.
2. **Validate** — dataset checks detect missing targets, duplicate IDs, missingness, and grouping problems before training.
3. **Learn** — preprocessing and estimators live inside pipelines so transformations are fit on training data only.
4. **Evaluate** — validation is used for model selection; the final test set is reserved for frozen-model assessment. Cross-validation, calibration, permutation importance, and benchmark tables are available.
5. **Export** — run manifests, dataset cards, fingerprints, and prediction exports provide a reproducible handoff to CardiEval and other components.

## Core boundary

The most important scientific boundary is the split. A patient, donor, animal, experiment, or study must not contribute related observations to both training and evaluation. Longitudinal data can use forward-chaining time splits rather than random splits.

## Modality strategy

CardiLearn does not force every modality into the same representation. Omics, ECG/time-series, imaging, and tabular data each get a native input contract, then converge into model-ready features or embeddings. Future deep encoders should implement the same stable interface and remain subject to the same split and provenance rules.

## Ecosystem boundary

CardiLearn owns learning and model development. CardiBench owns canonical datasets/splits. CardiEval independently evaluates exported predictions. CardiTrace records provenance across the wider system. This separation reduces optimistic self-evaluation and makes model comparisons reproducible.
