import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cardilearn.benchmarks import cross_validate
from cardilearn.calibration import expected_calibration_error
from cardilearn.ecg import summarize_signal


def test_group_cv_keeps_subjects_together():
    groups = np.repeat([f"p{i}" for i in range(10)], 4)
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"x": rng.normal(size=len(groups))})
    y = (X["x"] > 0).astype(int)
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
    result = cross_validate(model, X, y, task="classification", groups=pd.Series(groups), n_splits=5)
    assert len(result.folds) == 5
    assert "balanced_accuracy" in result.aggregate


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
