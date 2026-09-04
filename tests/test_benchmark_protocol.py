from __future__ import annotations

import numpy as np
import pytest

from cardilearn.benchmark_protocol import (
    BenchmarkSpec,
    compare_seeded_scores,
    rank_models,
    summarize_repeated_scores,
    validate_benchmark_record,
)


def test_spec_validation_and_defaults():
    spec = BenchmarkSpec(benchmark_id="synthetic_v1", task="classification", primary_metric="auroc", primary_direction="maximize")
    spec.validate()
    assert spec.models == ("pca_linear", "mlp", "autoencoder", "cardilearn")
    assert len(spec.seeds) == 5


def test_seeded_paired_comparison():
    result = compare_seeded_scores(
        {"cardilearn": [0.90, 0.91, 0.89, 0.92], "pca_linear": [0.70, 0.72, 0.71, 0.69]},
        primary_model="cardilearn",
        alternative="pca_linear",
    )
    assert result["n_seeds"] == 4
    assert result["mean_delta"] > 0
    assert 0 <= result["wilcoxon_pvalue"] <= 1


def test_ranking_direction():
    assert rank_models({"a": 0.8, "b": 0.9}, direction="maximize")[0] == ("b", 0.9)
    assert rank_models({"a": 0.2, "b": 0.1}, direction="minimize")[0] == ("b", 0.1)


def test_summary_and_record_validation():
    summary = summarize_repeated_scores([1.0, 2.0, 3.0])
    assert summary["n"] == 3
    assert np.isclose(summary["mean"], 2.0)
    validate_benchmark_record({"model": "cardilearn", "split": "test", "metric": "auroc", "value": 0.9, "seed": 42, "benchmark_id": "b1"})
    with pytest.raises(ValueError):
        validate_benchmark_record({"model": "cardilearn", "split": "validation", "metric": "auroc", "value": 0.9, "benchmark_id": "b1"})
