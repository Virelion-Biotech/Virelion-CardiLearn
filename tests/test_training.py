import numpy as np
import pandas as pd

from cardilearn.config import SplitConfig, TrainingConfig
from cardilearn.data import Dataset
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
