# CardiLearn Scientific Validation Readiness

This checklist separates **software correctness** from **empirical scientific validation**. CardiLearn should not report benchmark performance as a scientific result until the applicable items below have been completed on the real source cohorts.

## 1. Dataset provenance

- [ ] Source accession and publication recorded.
- [ ] Raw source material downloaded from the authoritative repository.
- [ ] SHA-256 or equivalent content hash recorded.
- [ ] GEO/SRA/sample metadata parsed without manual guessing.
- [ ] Biological sample, donor/animal, study, treatment, timepoint, and technical identifiers reconciled.
- [ ] Inclusion/exclusion rules written before model fitting.
- [ ] Any unresolved metadata ambiguity causes a validation failure rather than silent imputation.

## 2. Biological unit of inference

- [ ] Independent biological replicate explicitly identified.
- [ ] Cell-level observations are linked to their biological sample.
- [ ] No biological sample crosses train/validation/test boundaries.
- [ ] Pseudobulk/sample-level analysis is used for primary inferential claims when appropriate.
- [ ] Confidence intervals resample independent biological units rather than individual cells.

## 3. Feature construction

- [ ] Feature schema is frozen before held-out evaluation.
- [ ] Normalization and preprocessing are fit only on training data where applicable.
- [ ] Supervised feature selection is performed only inside development/training boundaries.
- [ ] PCA/representation learning is fit only on training data for predictive evaluation.
- [ ] No post-test feature filtering or manual feature selection is performed.

## 4. Model development

- [ ] Baselines include a simple non-learning reference where meaningful.
- [ ] Candidate models are evaluated under the same split contract.
- [ ] Hyperparameter/model selection uses development data only.
- [ ] Random seeds and configuration are recorded.
- [ ] Model artifacts can be reconstructed from the recorded configuration and data fingerprint.

## 5. Internal evaluation

- [ ] Group-aware cross-validation is feasible for the number of biological groups.
- [ ] Every fold contains the required outcome classes where the metric requires them.
- [ ] AUROC/AUPRC and threshold-dependent metrics are reported with appropriate uncertainty.
- [ ] Calibration is assessed for probabilistic predictions.
- [ ] Permutation/null analyses are used where they answer a defined scientific question.

## 6. External validation

- [ ] External cohort is locked before final model evaluation.
- [ ] No external labels or features influence model selection.
- [ ] Cross-study feature harmonization is defined before evaluation.
- [ ] Training-only feature selection is preserved during transfer.
- [ ] Performance and uncertainty are reported without retuning on the external cohort.
- [ ] Study/batch effects are explicitly assessed rather than interpreted as biology by default.

## 7. Interpretation

- [ ] Feature importance is calculated on validation/development data, not the held-out test set for exploratory tuning.
- [ ] Important features are checked for technical/batch confounding.
- [ ] Biological interpretation is separated from predictive association.
- [ ] No causal claim is made from predictive importance alone.

## 8. Publication package

A release should include, where permitted by data licenses:

- benchmark definition;
- source manifest and hashes;
- split manifest;
- feature schema;
- model configuration;
- model-selection record;
- frozen predictions;
- metrics and confidence intervals;
- calibration results;
- uncertainty/null analyses;
- environment/package versions;
- limitations and failure cases.

## Current CardiLearn position

The repository currently contains substantial infrastructure for items in sections 2–5 and cross-study transfer/uncertainty machinery. The remaining publication-critical work is primarily **real-data materialization and execution**, followed by independent validation and reporting. Do not substitute synthetic fixtures or unit tests for those empirical steps.
