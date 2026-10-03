"""Optional advanced evaluation utilities built on scikit-learn."""
from __future__ import annotations
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score

def cross_validated_scores(model, X, y, *, groups=None, cv=5, scoring="roc_auc") -> dict[str, float]:
    if groups is None:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring)
    else:
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=cv)
        scores = cross_val_score(model, X, y, cv=splitter, groups=groups, scoring=scoring)
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=1)), "scores": scores.tolist()}

def calibrate_classifier(model, X_train, y_train, method="sigmoid"):
    calibrated = CalibratedClassifierCV(model, method=method, cv=5)
    return calibrated.fit(X_train, y_train)

def permutation_importance_table(model, X, y, *, scoring="roc_auc", n_repeats=10):
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=42)
    names = getattr(X, "columns", [f"feature_{i}" for i in range(X.shape[1])])
    order = np.argsort(result.importances_mean)[::-1]
    return [{"feature": str(names[i]), "importance_mean": float(result.importances_mean[i]), "importance_std": float(result.importances_std[i])} for i in order]
