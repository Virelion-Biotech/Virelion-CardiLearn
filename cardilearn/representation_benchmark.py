"""Independent representation benchmarking for CardiLearn embeddings.

The benchmark operates on frozen embeddings and never mutates model weights.
It supports biological-group-aware cross-validation, permutation nulls, and
bootstrap uncertainty. It deliberately returns fold-level records so pooled
metrics cannot conceal weak biological generalization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    GroupKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FoldResult:
    model: str
    fold: int
    n_train: int
    n_test: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class BenchmarkSummary:
    model: str
    task: str
    primary_metric: str
    mean: float
    std: float
    fold_results: tuple[FoldResult, ...]


def _validate_embeddings(z: np.ndarray, n: int) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != n:
        raise ValueError("embeddings must be [n_observations, dimensions]")
    if not np.isfinite(z).all():
        raise ValueError("embeddings contain non-finite values")
    return z


def _classification_metrics(y_true, prediction, score) -> dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1_macro": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        result["auroc"] = float(roc_auc_score(y_true, score))
        result["auprc"] = float(average_precision_score(y_true, score))
    return result


def _make_splitter(
    task: str,
    n_splits: int,
    *,
    groups: np.ndarray | None,
    y: np.ndarray,
    seed: int,
):
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if groups is not None:
        if len(np.unique(groups)) < n_splits:
            raise ValueError("not enough biological groups for requested folds")
        if task == "classification":
            counts = pd.DataFrame({"y": y, "g": groups}).groupby("y")["g"].nunique()
            if (counts < n_splits).any():
                raise ValueError(
                    "each class needs at least n_splits biological groups for stratified grouped CV"
                )
            return StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
        return GroupKFold(n_splits=n_splits)
    if task == "classification":
        if np.min(np.bincount(pd.factorize(y)[0])) < n_splits:
            raise ValueError("each class needs at least n_splits observations")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def benchmark_embedding(
    z: np.ndarray,
    y: Iterable,
    *,
    model_name: str,
    task: str,
    primary_metric: str | None = None,
    groups: Iterable | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> BenchmarkSummary:
    labels = np.asarray(list(y))
    if labels.ndim != 1:
        raise ValueError("y must be one-dimensional")
    embeddings = _validate_embeddings(z, len(labels))
    group_values = None if groups is None else np.asarray(list(groups), dtype=object)
    if group_values is not None and len(group_values) != len(labels):
        raise ValueError("groups and y must have identical lengths")
    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression")

    splitter = _make_splitter(
        task, n_splits, groups=group_values, y=labels, seed=seed
    )
    splits = splitter.split(embeddings, labels, group_values) if group_values is not None else splitter.split(embeddings, labels)

    folds: list[FoldResult] = []
    for fold_index, (train_idx, test_idx) in enumerate(splits):
        if task == "classification":
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=seed),
            )
            estimator.fit(embeddings[train_idx], labels[train_idx])
            prediction = estimator.predict(embeddings[test_idx])
            score = estimator.predict_proba(embeddings[test_idx])[:, 1]
            metrics = _classification_metrics(labels[test_idx], prediction, score)
            metric = primary_metric or "auroc"
        else:
            target = labels.astype(float)
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            estimator.fit(embeddings[train_idx], target[train_idx])
            prediction = estimator.predict(embeddings[test_idx])
            metrics = {
                "mae": float(mean_absolute_error(target[test_idx], prediction)),
                "rmse": float(np.sqrt(mean_squared_error(target[test_idx], prediction))),
                "r2": float(r2_score(target[test_idx], prediction)),
            }
            metric = primary_metric or "rmse"
        if metric not in metrics:
            raise ValueError(f"primary metric '{metric}' is unavailable for fold {fold_index}")
        folds.append(
            FoldResult(
                model=model_name,
                fold=fold_index,
                n_train=len(train_idx),
                n_test=len(test_idx),
                metrics=metrics,
            )
        )

    values = np.asarray([fold.metrics[metric] for fold in folds], dtype=float)
    return BenchmarkSummary(
        model=model_name,
        task=task,
        primary_metric=metric,
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        fold_results=tuple(folds),
    )


def benchmark_embedding_matrix(
    embeddings: Mapping[str, np.ndarray],
    y: Iterable,
    **kwargs,
) -> tuple[BenchmarkSummary, ...]:
    labels = list(y)
    results = []
    for name, values in embeddings.items():
        results.append(
            benchmark_embedding(values, labels, model_name=name, **kwargs)
        )
    return tuple(results)


def paired_seed_deltas(
    summaries: Mapping[str, Iterable[BenchmarkSummary]],
    *,
    reference: str,
    alternative: str,
) -> np.ndarray:
    if reference not in summaries or alternative not in summaries:
        raise KeyError("reference and alternative must both be present")
    ref = list(summaries[reference])
    alt = list(summaries[alternative])
    if len(ref) != len(alt) or not ref:
        raise ValueError("seed summary collections must have equal non-zero length")
    if any(a.primary_metric != b.primary_metric for a, b in zip(ref, alt)):
        raise ValueError("paired summaries must use the same primary metric")
    return np.asarray([a.mean - b.mean for a, b in zip(ref, alt)], dtype=float)


def permutation_null_auroc(
    z: np.ndarray,
    y: Iterable,
    *,
    groups: Iterable | None,
    n_permutations: int = 200,
    seed: int = 42,
    n_splits: int = 5,
) -> dict[str, float | int]:
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    labels = np.asarray(list(y))
    observed = benchmark_embedding(
        z,
        labels,
        model_name="observed",
        task="classification",
        primary_metric="auroc",
        groups=groups,
        n_splits=n_splits,
        seed=seed,
    ).mean
    rng = np.random.default_rng(seed)
    null_values = []
    group_values = None if groups is None else np.asarray(list(groups), dtype=object)
    for _ in range(n_permutations):
        shuffled = labels.copy()
        if group_values is None:
            rng.shuffle(shuffled)
        else:
            unique_groups = np.unique(group_values)
            group_labels = {group: labels[np.where(group_values == group)[0][0]] for group in unique_groups}
            permuted_values = list(group_labels.values())
            rng.shuffle(permuted_values)
            mapping = dict(zip(unique_groups, permuted_values))
            shuffled = np.asarray([mapping[g] for g in group_values], dtype=labels.dtype)
        try:
            value = benchmark_embedding(
                z,
                shuffled,
                model_name="null",
                task="classification",
                primary_metric="auroc",
                groups=group_values,
                n_splits=n_splits,
                seed=seed,
            ).mean
        except ValueError:
            continue
        null_values.append(value)
    if not null_values:
        raise ValueError("no valid permutation replicates")
    null = np.asarray(null_values, dtype=float)
    return {
        "observed_auroc": float(observed),
        "null_mean_auroc": float(null.mean()),
        "null_std_auroc": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        "n_valid_permutations": int(len(null)),
        "p_value_one_sided": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
    }


def bootstrap_metric(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    metric: str = "auroc",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | list[float]]:
    y = np.asarray(list(y_true))
    score = np.asarray(list(y_score), dtype=float)
    if y.shape != score.shape:
        raise ValueError("y_true and y_score must have matching shapes")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y), len(y))
        sample_y = y[indices]
        if metric == "auroc":
            if len(np.unique(sample_y)) < 2:
                continue
            values.append(float(roc_auc_score(sample_y, score[indices])))
        elif metric == "auprc":
            values.append(float(average_precision_score(sample_y, score[indices])))
        else:
            raise ValueError("metric must be 'auroc' or 'auprc'")
    if not values:
        raise ValueError("no valid bootstrap replicates")
    interval = np.quantile(values, [0.025, 0.975])
    return {
        "n_valid": len(values),
        "mean": float(np.mean(values)),
        "ci95_low": float(interval[0]),
        "ci95_high": float(interval[1]),
    }


def summary_to_dict(summary: BenchmarkSummary) -> dict[str, object]:
    return asdict(summary)
