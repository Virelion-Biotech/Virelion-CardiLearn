from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cardilearn.cross_species import (
    evaluate_classification_transfer,
    evaluate_regression_transfer,
    leave_one_species_out,
    summarize_transfer,
    trajectory_conservation_score,
)


def metadata() -> pd.DataFrame:
    rows = []
    for species, study, subject_prefix in [("human", "H1", "h"), ("mouse", "M1", "m"), ("pig", "P1", "p")]:
        for i in range(4):
            rows.append(
                {
                    "species": species,
                    "study_id": study,
                    "subject_id": f"{subject_prefix}{i}",
                    "sample_id": f"{subject_prefix}{i}s",
                }
            )
    return pd.DataFrame(rows)


def test_leave_one_species_out_is_deterministic_and_exhaustive():
    frame = metadata()
    splits = leave_one_species_out(frame)
    assert [split.held_out_species for split in splits] == ["human", "mouse", "pig"]
    for split in splits:
        assert len(set(split.train_indices).intersection(split.test_indices)) == 0
        assert len(split.train_indices) + len(split.test_indices) == len(frame)
        assert len(split.test_indices) == 4


def test_leave_one_species_out_requires_identity_metadata():
    frame = metadata().drop(columns=["subject_id"])
    with pytest.raises(ValueError, match="missing leakage-audit columns"):
        leave_one_species_out(frame)


def test_classification_transfer_uses_frozen_embeddings_only():
    rng = np.random.default_rng(7)
    frame = metadata()
    Z = rng.normal(size=(len(frame), 6)).astype(np.float32)
    y = np.array([0, 1, 0, 1] * 3)
    split = leave_one_species_out(frame)[0]
    result = evaluate_classification_transfer(Z, y, split)
    assert result["task"] == "classification"
    assert result["held_out_species"] == "human"
    assert 0.0 <= result["balanced_accuracy"] <= 1.0


def test_regression_transfer_reports_r2_and_mae():
    frame = metadata()
    Z = np.arange(len(frame) * 2, dtype=np.float32).reshape(len(frame), 2)
    y = Z[:, 0] * 0.5 + 2.0
    result = evaluate_regression_transfer(Z, y, leave_one_species_out(frame)[1])
    assert result["task"] == "regression"
    assert "r2" in result and "mae" in result


def test_trajectory_conservation_is_high_for_monotone_latent_axis():
    rows = []
    for species, offset in [("human", 0.0), ("mouse", 10.0), ("pig", 20.0)]:
        for stage in range(5):
            rows.append((species, float(stage), np.array([stage + offset, 0.1 * stage, 0.0])))
    species, stage, embeddings = zip(*rows)
    result = trajectory_conservation_score(np.asarray(embeddings), stage, species)
    assert result["median_abs_spearman"] > 0.95
    assert result["species_count_evaluated"] == 3.0


def test_summarize_transfer_averages_numeric_metrics():
    results = [
        {"held_out_species": "human", "balanced_accuracy": 0.8, "auroc": 0.9},
        {"held_out_species": "mouse", "balanced_accuracy": 0.6, "auroc": 0.7},
    ]
    summary = summarize_transfer(results)
    assert summary == {"auroc": 0.8, "balanced_accuracy": 0.7}
