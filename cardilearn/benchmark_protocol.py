"""Step 15: leakage-safe benchmark protocol for CardiLearn and baselines.

This module defines the comparison contract; it does not claim any model wins.
Real benchmark results require locked datasets, declared splits and completed runs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import wilcoxon


DEFAULT_MODELS = (
    "pca_linear",
    "mlp",
    "autoencoder",
    "cardilearn",
)


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    task: str
    primary_metric: str
    primary_direction: str
    models: tuple[str, ...] = DEFAULT_MODELS
    seeds: tuple[int, ...] = (13, 42, 73, 101, 131)
    selection_split: str = "validation"
    report_split: str = "test"
    require_locked_test: bool = True
    require_grouped_split: bool = True

    def validate(self) -> None:
        if self.task not in {"classification", "regression", "representation"}:
            raise ValueError(f"unsupported task: {self.task}")
        if self.primary_direction not in {"maximize", "minimize"}:
            raise ValueError("primary_direction must be maximize or minimize")
        if not self.models:
            raise ValueError("at least one model is required")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")

    def to_dict(self) -> dict:
        return asdict(self)


def compare_seeded_scores(
    scores: Mapping[str, Iterable[float]],
    *,
    primary_model: str = "cardilearn",
    alternative: str | None = None,
) -> dict[str, float | int | str]:
    """Paired seeded comparison; seeds must align by position.

    The Wilcoxon test is reported without implying statistical significance from
    the tiny default seed count. This is a comparison utility, not a publication claim.
    """
    if primary_model not in scores:
        raise KeyError(primary_model)
    candidate = alternative or next(name for name in scores if name != primary_model)
    a = np.asarray(tuple(scores[primary_model]), dtype=float)
    b = np.asarray(tuple(scores[candidate]), dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired score vectors must have identical length")
    if a.size < 2:
        raise ValueError("at least two paired scores are required")
    delta = a - b
    try:
        statistic, p_value = wilcoxon(delta)
    except ValueError:
        statistic, p_value = float("nan"), 1.0
    return {
        "primary_model": primary_model,
        "alternative_model": candidate,
        "n_seeds": int(a.size),
        "mean_primary": float(a.mean()),
        "mean_alternative": float(b.mean()),
        "mean_delta": float(delta.mean()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_pvalue": float(p_value),
    }


def rank_models(
    scores: Mapping[str, float],
    *,
    direction: str,
) -> list[tuple[str, float]]:
    """Rank models on a single locked evaluation set."""
    if direction == "maximize":
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if direction == "minimize":
        return sorted(scores.items(), key=lambda item: item[1])
    raise ValueError("direction must be maximize or minimize")


def validate_benchmark_record(record: Mapping[str, object]) -> None:
    """Reject result records that could silently mix validation/test selection."""
    required = {"model", "split", "metric", "value", "seed", "benchmark_id"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"benchmark record missing fields: {sorted(missing)}")
    if record["split"] not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    if not isinstance(record["seed"], int):
        raise TypeError("seed must be int")


def summarize_repeated_scores(scores: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(tuple(scores), dtype=float)
    if values.size == 0:
        raise ValueError("scores cannot be empty")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }
