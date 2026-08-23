"""Stable interoperability formats for downstream evaluation and provenance tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PREDICTION_FORMAT = "cardilearn.predictions.v1"


def export_predictions(
    sample_ids,
    predictions,
    *,
    output: str | Path,
    probabilities=None,
    split: str = "test",
    model_name: str = "unknown",
    dataset_fingerprint: str | None = None,
) -> Path:
    """Write the legacy-compatible tidy prediction table."""
    frame = pd.DataFrame({"sample_id": list(sample_ids), "prediction": list(predictions)})
    frame["split"] = split
    frame["model_name"] = model_name
    if probabilities is not None:
        probs = probabilities
        if getattr(probs, "ndim", 1) == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]
        frame["probability"] = list(probs)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    meta = out.with_suffix(out.suffix + ".json")
    meta.write_text(
        json.dumps(
            {
                "format": PREDICTION_FORMAT,
                "model_name": model_name,
                "split": split,
                "dataset_fingerprint": dataset_fingerprint,
                "n_predictions": len(frame),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return out


def export_frozen_predictions(
    *,
    sample_ids,
    observation_ids,
    true_labels,
    predicted_labels,
    probabilities,
    folds,
    dataset,
    model,
    output: str | Path,
    dataset_fingerprint: str | None = None,
) -> Path:
    """Export the stable `cardilearn.predictions.v1` CardiEval interchange table."""
    values = [sample_ids, observation_ids, true_labels, predicted_labels, probabilities, folds]
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError("all prediction fields must have the same length")
    frame = pd.DataFrame(
        {
            "sample_id": list(sample_ids),
            "observation_id": list(observation_ids),
            "true_label": list(true_labels),
            "predicted_label": list(predicted_labels),
            "probability": list(probabilities),
            "fold": list(folds),
            "dataset": dataset,
            "model": model,
        }
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    out.with_suffix(out.suffix + ".json").write_text(
        json.dumps(
            {
                "format": PREDICTION_FORMAT,
                "dataset": dataset,
                "model": model,
                "dataset_fingerprint": dataset_fingerprint,
                "n_predictions": len(frame),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return out


def load_predictions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_id", "prediction", "split", "model_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction file missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("prediction file contains duplicate sample IDs")
    return frame
