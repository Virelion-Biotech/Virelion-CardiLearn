"""Leakage-aware hyperparameter search utilities."""
from __future__ import annotations

from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold


def grid_search(
    estimator,
    X,
    y,
    param_grid,
    *,
    task="classification",
    groups=None,
    cv=5,
    scoring=None,
):
    if cv < 2:
        raise ValueError("cv must be >= 2")
    if groups is not None:
        if task == "classification":
            splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            splitter = GroupKFold(n_splits=cv)
    else:
        splitter = (
            StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            if task == "classification"
            else KFold(n_splits=cv, shuffle=True, random_state=42)
        )
    score = scoring or ("roc_auc" if task == "classification" else "neg_root_mean_squared_error")
    search = GridSearchCV(estimator, param_grid=param_grid, cv=splitter, scoring=score, n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)
    return search
