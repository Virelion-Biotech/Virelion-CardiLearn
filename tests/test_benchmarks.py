import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cardilearn.benchmarks import cross_validate
from cardilearn.calibration import expected_calibration_error
from cardilearn.ecg import summarize_signal
from cardilearn.splitting import make_classification_splitter


def test_stratified_group_cv_keeps_samples_together_and_classes_present():
    groups = np.repeat(["sample_1", "sample_2", "sample_3", "sample_4"], 5)
    y = np.repeat([0, 0, 1, 1], 5)
    X = pd.DataFrame({"x": np.where(y == 1, 1.0, -1.0) + np.linspace(0, 0.1, len(y))})
    splitter = make_classification_splitter(2, groups, y, random_state=42)

    for train_idx, validation_idx in splitter.split(X, y, groups):
        train_groups = set(groups[train_idx])
        validation_groups = set(groups[validation_idx])
        assert train_groups.isdisjoint(validation_groups)
        assert set(y[validation_idx]) == {0, 1}

        model = LogisticRegression().fit(X.iloc[train_idx], pd.Series(y).iloc[train_idx])
        probabilities = model.predict_proba(X.iloc[validation_idx])[:, 1]
        assert np.isfinite(roc_auc_score(y[validation_idx], probabilities))


def test_stratified_group_cv_is_used_by_benchmark_classification():
    groups = np.repeat([f"sample_{i}" for i in range(6)], 3)
    y = np.repeat([0, 0, 0, 1, 1, 1], 3)
    X = pd.DataFrame({"x": y + np.linspace(-0.2, 0.2, len(y))})
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
    result = cross_validate(
        model,
        X,
        pd.Series(y),
        task="classification",
        groups=pd.Series(groups),
        n_splits=3,
    )
    assert len(result.folds) == 3
    assert all(np.isfinite(f.metrics["auroc"]) for f in result.folds)


def test_stratified_group_cv_rejects_infeasible_class_group_counts():
    groups = np.repeat(["sham_1", "sham_2", "mi_1", "mi_2"], 2)
    y = np.repeat([0, 0, 1, 1], 2)
    with pytest.raises(ValueError, match="each class needs at least 3 biological groups"):
        make_classification_splitter(3, groups, y)


def test_stratified_group_cv_rejects_non_binary_targets():
    groups = np.repeat(["a", "b", "c"], 2)
    y = np.repeat([0, 1, 2], 2)
    with pytest.raises(ValueError, match="exactly two classes"):
        make_classification_splitter(2, groups, y)


def test_existing_generic_group_cv_remains_available_for_regression():
    groups = np.repeat([f"p{i}" for i in range(10)], 4)
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"x": rng.normal(size=len(groups))})
    y = X["x"] * 2.0
    from sklearn.linear_model import Ridge

    result = cross_validate(
        Ridge(), X, y, task="regression", groups=pd.Series(groups), n_splits=5
    )
    assert len(result.folds) == 5


def test_ecg_summary_has_expected_signal_features():
    signal = np.sin(np.linspace(0, 4 * np.pi, 200))
    features = summarize_signal(signal, sampling_rate_hz=200)
    assert features["n_samples"] == 200
    assert features["duration_s"] > 0
    assert features["range"] > 1.0


def test_ece_is_bounded():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert 0.0 <= expected_calibration_error(y, p) <= 1.0
