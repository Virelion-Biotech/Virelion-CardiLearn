"""Model selection helpers that never use the final test partition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pandas as pd

from .benchmarks import cross_validate


@dataclass(frozen=True)
class CandidateScore:
    name: str
    aggregate: dict[str, float]
    variability: dict[str, float]


def select_model(
    candidates: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: str,
    primary_metric: str,
    groups: pd.Series | None = None,
    n_splits: int = 5,
    higher_is_better: bool = True,
) -> tuple[str, tuple[CandidateScore, ...]]:
    """Select a candidate using CV on the development data only."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    scores: list[CandidateScore] = []
    for name, model in candidates.items():
        cv = cross_validate(model, X, y, task=task, groups=groups, n_splits=n_splits)
        if primary_metric not in cv.aggregate:
            raise KeyError(f"metric '{primary_metric}' unavailable for candidate '{name}'")
        scores.append(CandidateScore(name, cv.aggregate, cv.variability))
    scores.sort(key=lambda s: s.aggregate[primary_metric], reverse=higher_is_better)
    return scores[0].name, tuple(scores)
