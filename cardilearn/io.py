"""Stable interoperability formats for downstream evaluation and provenance tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


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
    """Write a tidy prediction table that CardiEval can ingest."""
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
                "format": "cardilearn.predictions.v1",
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


def load_predictions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_id", "prediction", "split", "model_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction file missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("prediction file contains duplicate sample IDs")
    return frame
