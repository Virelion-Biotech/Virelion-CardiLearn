# CardiLearn Experiment Protocol

This protocol defines the minimum standard for a publishable CardiLearn experiment.

## 1. Dataset registration

Record the dataset source, accession or persistent identifier, species, tissue, modality, assay technology, sample IDs, subject IDs, collection time, intervention/condition, and preprocessing provenance before model training.

Do not mix biological replicates with technical replicates as independent subjects. When multiple observations belong to the same patient, donor, animal, or experiment, encode the appropriate `group_id` and keep that group inside one partition.

## 2. Target definition

State the prediction target exactly, including its time horizon and whether it is a classification, regression, ordinal, survival, or multilabel task. Exclude post-outcome variables and identifiers that leak the target.

## 3. Partitioning

Create train, validation, and final test partitions before tuning. Prefer group-aware splits for repeated measurements and study-aware splits when external-study generalization is the question. For longitudinal tasks, use temporal ordering when future observations should represent deployment.

## 4. Preprocessing

Fit imputation, normalization, supervised feature selection, dimensionality reduction, and learned representations on the training partition only. Freeze those transforms before validation/test prediction.

## 5. Model development

Establish a simple baseline first. Compare stronger candidates using cross-validation inside the development partition. Select one model using a prespecified primary metric. Do not rank candidates by final test performance.

## 6. Calibration and explainability

For probabilistic classifiers, report discrimination and calibration separately. When feature attribution is used, compute it on a validation/development set and retain the exact model and feature schema used.

## 7. Final evaluation

After model selection, evaluate the frozen model once on the final test partition. Export predictions with sample IDs, split identity, model name, and dataset fingerprint. CardiEval should consume those frozen predictions for independent statistical comparison.

## 8. External validation

The strongest next step is a dataset not used in model development, ideally from a distinct study, laboratory, site, or acquisition platform. Report performance degradation and calibration shift rather than only pooled performance.

## 9. Reporting

A complete run should retain:

- dataset and model provenance
- exact configuration
- dataset fingerprint
- feature schema
- split assignments
- development CV results
- final test predictions and metrics
- calibration metrics when applicable
- important implementation/version metadata

A model is not considered benchmark-ready if these artifacts cannot be reproduced.
