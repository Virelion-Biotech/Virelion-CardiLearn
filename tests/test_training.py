import numpy as np
import pandas as pd

from cardilearn.config import SplitConfig, TrainingConfig
from cardilearn.data import Dataset
from cardilearn.metrics import classification_metrics, regression_metrics
from cardilearn.training import evaluate_held_out_test, train


def _classification_dataset():
    rng = np.random.default_rng(7)
    groups = np.repeat([f"patient_{i}" for i in range(12)], 3)
    x1 = rng.normal(size=36)
    x2 = rng.normal(size=36)
    target = (x1 + 0.5 * x2 > 0).astype(int)
    return Dataset(pd.DataFrame({"x1": x1, "x2": x2, "target": target, "group_id": groups}), "target", "group_id")


def test_training_does_not_touch_test_metrics():
    dataset = _classification_dataset()
    config = TrainingConfig(task="classification", model="logistic_regression", target_column="target", group_column="group_id", split=SplitConfig(random_state=3))
    result = train(dataset, config)
    assert set(result.metrics) == {"train", "validation"}
    assert result.splits.train.size + result.splits.validation.size + result.splits.test.size == 36
    final = evaluate_held_out_test(result, dataset, "classification")
    assert "balanced_accuracy" in final



def test_undefined_classification_auroc_is_explicitly_null_and_json_safe():
    metrics = classification_metrics(
        np.asarray([1, 1, 1]),
        np.asarray([1, 1, 1]),
        np.asarray([0.8, 0.9, 0.7]),
    )
    assert metrics["auroc"] is None
    import json

    assert '"auroc": null' in json.dumps(metrics, sort_keys=True, allow_nan=False)


def test_undefined_regression_r2_is_explicitly_null_and_json_safe():
    metrics = regression_metrics(
        np.asarray([2.0]),
        np.asarray([2.0]),
    )
    assert metrics["r2"] is None
    import json

    assert '"r2": null' in json.dumps(metrics, sort_keys=True, allow_nan=False)
