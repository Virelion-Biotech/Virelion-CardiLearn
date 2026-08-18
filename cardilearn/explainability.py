"""Model-agnostic, dependency-light feature importance helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_frame(
    estimator,
    X: pd.DataFrame,
    y,
    *,
    scoring: str | None = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on a supplied validation set.

    This function does not fit or tune the estimator and therefore cannot accidentally
    convert a held-out test partition into a feature-selection loop.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    result = permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
