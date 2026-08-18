import pandas as pd

from cardilearn.benchmark_matrix import run_model_matrix
from cardilearn.cardiobench import BenchmarkDefinition


def test_mi_matrix_runs_on_prepared_table():
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "biological_group_id": [f"g{i}" for i in range(12)],
            "injury_label": ["MI", "sham"] * 6,
            "x1": [float(i) for i in range(12)],
            "x2": [float(i % 3) for i in range(12)],
        }
    )
    definition = BenchmarkDefinition(
        benchmark_id="test-mi-sham",
        version="1",
        task="classification",
        target="injury_label",
        split_policy="subject",
        group_key="biological_group_id",
    )
    results = run_model_matrix(frame, definition, models=["logistic_regression"])
    assert len(results) == 1
    assert results[0].test is None
    assert "balanced_accuracy" in results[0].validation
