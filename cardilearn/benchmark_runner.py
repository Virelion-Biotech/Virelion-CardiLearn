"""Run a single CardiBench-compatible benchmark on a prepared feature table."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .cardiobench import BenchmarkDefinition, load_definition, validate_definition_against_frame
from .config import SplitConfig, TrainingConfig
from .data import Dataset
from .models import build_model
from .training import TrainingResult
from .splitting import split_frame
from .metrics import evaluate
from .reproducibility import dataframe_fingerprint


@dataclass(frozen=True)
class BenchmarkRun:
    definition: BenchmarkDefinition
    result: TrainingResult
    source_path: str


def run_benchmark(
    data_path: str | Path,
    definition_path: str | Path,
    *,
    model_name: str = "logistic_regression",
    seed: int = 42,
) -> BenchmarkRun:
    """Train a benchmark model using the definition's target/group contract.

    The first implementation supports subject/donor/animal/technical-group style
    internal splits. Study/timepoint/species held-out protocols must be materialized
    explicitly by CardiBench before training; they are never inferred from arbitrary
    column values here.
    """
    definition = load_definition(definition_path)
    frame = pd.read_csv(data_path)
    validate_definition_against_frame(definition, frame)

    if definition.split_policy not in {
        "subject", "donor", "animal", "technical_group", "sample_with_warning"
    }:
        raise ValueError(
            f"split policy '{definition.split_policy}' requires an explicit CardiBench split manifest"
        )

    dataset = Dataset(frame, target_column=definition.target, group_column=definition.group_key)
    config = TrainingConfig(
        task=definition.task,
        model=model_name,
        target_column=definition.target,
        group_column=definition.group_key,
        random_state=seed,
        split=SplitConfig(random_state=seed),
    )
    # Inline training keeps the benchmark result tied to the benchmark definition.
    splits = split_frame(frame, config.split, target=dataset.target, groups=dataset.groups)
    X, y = dataset.features(), dataset.target
    model = build_model(definition.task, model_name, X.iloc[splits.train])
    model.fit(X.iloc[splits.train], y.iloc[splits.train])
    metrics = {
        "train": evaluate(model, X.iloc[splits.train], y.iloc[splits.train], definition.task),
        "validation": evaluate(model, X.iloc[splits.validation], y.iloc[splits.validation], definition.task),
    }
    result = TrainingResult(
        model=model,
        splits=splits,
        metrics=metrics,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        group_column=dataset.group_column,
        elapsed_seconds=0.0,
        dataset_fingerprint=dataframe_fingerprint(frame),
    )
    return BenchmarkRun(definition, result, str(data_path))
