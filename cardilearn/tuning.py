"""Leakage-aware hyperparameter search utilities."""
from __future__ import annotations
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold

def grid_search(estimator, X, y, param_grid, *, task="classification", groups=None, cv=5, scoring=None):
    splitter = GroupKFold(n_splits=cv) if groups is not None else StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    kwargs = {"cv": splitter, "scoring": scoring or ("roc_auc" if task == "classification" else "neg_root_mean_squared_error"), "n_jobs": -1, "refit": True}
    search = GridSearchCV(estimator, param_grid=param_grid, **kwargs)
    search.fit(X, y, groups=groups)
    return search
