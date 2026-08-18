"""Portable run artifacts and model manifests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib

from .config import TrainingConfig
from .training import TrainingResult


def save_run(result: TrainingResult, config: TrainingConfig, output_dir: str | Path) -> Path:
    """Save model, metrics, split indices, and configuration as one run directory."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, output / "model.joblib")

    manifest: dict[str, Any] = {
        "cardilearn_version": "0.1.0",
        "config": config.to_dict(),
        "dataset": {
            "target_column": result.target_column,
            "group_column": result.group_column,
            "feature_columns": result.feature_columns,
        },
        "metrics": result.metrics,
        "split_sizes": {
            "train": int(len(result.splits.train)),
            "validation": int(len(result.splits.validation)),
            "test": int(len(result.splits.test)),
        },
        "splits": {
            "train": result.splits.train.tolist(),
            "validation": result.splits.validation.tolist(),
            "test": result.splits.test.tolist(),
        },
        "elapsed_seconds": result.elapsed_seconds,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output
