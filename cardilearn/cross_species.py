"""Cross-species validation utilities for CardiLearn.

The evaluator is deliberately independent of the neural network implementation so
it can validate frozen embeddings from PyTorch, NumPy, or external baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cardilearn.leakage import assert_no_leakage


@dataclass(frozen=True)
class SpeciesSplit:
    """One leave-one-species-out split definition."""

    held_out_species: str
    train_indices: np.ndarray
    test_indices: np.ndarray


def leave_one_species_out(
    metadata: pd.DataFrame,
    *,
    species_column: str = "species",
    split_column: str = "_split",
) -> list[SpeciesSplit]:
    """Create deterministic leave-one-species-out splits.

    Existing identity metadata are audited, but all samples from the held-out
    species are assigned to test and all other species to train. No validation
    partition is invented because this utility is explicitly an external-transfer
    evaluation.
    """
    if species_column not in metadata.columns:
        raise ValueError(f"missing species column: {species_column}")
    species = metadata[species_column].replace("", pd.NA)
    if species.isna().any():
        raise ValueError("missing species labels cannot be used for species transfer")

    audit_frame = metadata.copy()
    audit_frame[split_column] = "unassigned"
    # Audit identity fields without requiring a validation/test partition during
    # construction. The actual split is returned separately below.
    audit_frame[split_column] = "train"
    assert_no_leakage(audit_frame, split_column=split_column, require_all_partitions=False)

    labels = species.astype(str).to_numpy()
    unique_species = sorted(pd.unique(labels).tolist())
    if len(unique_species) < 2:
        raise ValueError("at least two species are required for leave-one-species-out")

    splits: list[SpeciesSplit] = []
    indices = np.arange(len(metadata), dtype=int)
    for held_out in unique_species:
        test_mask = labels == held_out
        splits.append(
            SpeciesSplit(
                held_out_species=held_out,
                train_indices=indices[~test_mask],
                test_indices=indices[test_mask],
            )
        )
    return splits


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"expected 2-D embeddings, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("embeddings contain non-finite values")
    return X


def evaluate_classification_transfer(
    Z: np.ndarray,
    y: Iterable[int | str],
    split: SpeciesSplit,
) -> dict[str, float | str]:
    """Fit a frozen-embedding linear classifier on non-held-out species."""
    Z = _as_2d_float(Z)
    labels = np.asarray(list(y))
    if len(labels) != len(Z):
        raise ValueError("y and Z have incompatible lengths")
    if len(np.unique(labels[split.train_indices])) < 2:
        raise ValueError("training portion must contain at least two classes")

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=0),
    )
    classifier.fit(Z[split.train_indices], labels[split.train_indices])
    prediction = classifier.predict(Z[split.test_indices])
    result: dict[str, float | str] = {
        "task": "classification",
        "held_out_species": split.held_out_species,
        "balanced_accuracy": float(
            balanced_accuracy_score(labels[split.test_indices], prediction)
        ),
    }

    classes = np.unique(labels[split.train_indices])
    if len(classes) == 2:
        probabilities = classifier.predict_proba(Z[split.test_indices])[:, 1]
        positive = classes[1]
        binary_target = (labels[split.test_indices] == positive).astype(int)
        if len(np.unique(binary_target)) == 2:
            result["auroc"] = float(roc_auc_score(binary_target, probabilities))
    return result


def evaluate_regression_transfer(
    Z: np.ndarray,
    y: Iterable[float],
    split: SpeciesSplit,
) -> dict[str, float | str]:
    """Fit a frozen-embedding linear regressor on non-held-out species."""
    Z = _as_2d_float(Z)
    target = np.asarray(list(y), dtype=np.float64)
    if len(target) != len(Z):
        raise ValueError("y and Z have incompatible lengths")
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    regressor.fit(Z[split.train_indices], target[split.train_indices])
    prediction = regressor.predict(Z[split.test_indices])
    return {
        "task": "regression",
        "held_out_species": split.held_out_species,
        "r2": float(r2_score(target[split.test_indices], prediction)),
        "mae": float(mean_absolute_error(target[split.test_indices], prediction)),
    }


def trajectory_conservation_score(
    Z: np.ndarray,
    stage: Iterable[float],
    species: Iterable[str],
) -> dict[str, float | str]:
    """Measure whether latent position preserves an ordered biological stage.

    The score is computed within each species using the first principal axis of
    the supplied embeddings, then summarized by the median absolute Spearman
    correlation. This is a descriptive sanity check, not a causal test.
    """
    Z = _as_2d_float(Z)
    stages = np.asarray(list(stage), dtype=float)
    species_values = np.asarray(list(species), dtype=object)
    if not (len(stages) == len(species_values) == len(Z)):
        raise ValueError("Z, stage, and species must have identical lengths")

    centered = Z - Z.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    latent_position = centered @ axis

    correlations: list[float] = []
    for label in sorted(pd.unique(species_values).tolist()):
        mask = species_values == label
        if mask.sum() < 3 or np.unique(stages[mask]).size < 2:
            continue
        rho = spearmanr(latent_position[mask], stages[mask]).statistic
        if np.isfinite(rho):
            correlations.append(abs(float(rho)))
    if not correlations:
        raise ValueError("at least one species needs >=3 observations with varied stages")
    return {
        "task": "trajectory_conservation",
        "median_abs_spearman": float(np.median(correlations)),
        "species_count_evaluated": float(len(correlations)),
    }


def summarize_transfer(results: Iterable[dict[str, float | str]]) -> dict[str, float]:
    """Average compatible numeric transfer metrics across held-out species."""
    rows = list(results)
    if not rows:
        raise ValueError("at least one result is required")
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (float, int))})
    return {key: float(np.mean([float(row[key]) for row in rows if key in row])) for key in keys}
