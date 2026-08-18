"""Execute a reproducible model matrix over prepared CardiBench feature tables."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .cardiobench import BenchmarkDefinition, validate_definition_against_frame
from .config import SplitConfig, TrainingConfig
from .data import Dataset
from .metrics import evaluate
from .models import build_model
from .splitting import split_frame


@dataclass(frozen=True)
class MatrixResult:
    dataset: str
    benchmark_id: str
    model: str
    validation: dict[str, float]
    test: dict[str, float] | None
    n_train: int
    n_validation: int
    n_test: int


def run_model_matrix(
    frame: pd.DataFrame,
    definition: BenchmarkDefinition,
    *,
    models: Iterable[str],
    random_state: int = 42,
) -> list[MatrixResult]:
    """Train candidate models on one prepared feature table.

    The function intentionally evaluates validation metrics only. A caller must
    explicitly provide an independent test frame to evaluate final performance.
    """
    validate_definition_against_frame(definition, frame)
    dataset = Dataset(frame, target_column=definition.target, group_column=definition.group_key)
    split = split_frame(
        frame,
        SplitConfig(random_state=random_state, stratify=False),
        target=dataset.target,
        groups=dataset.groups,
    )
    X, y = dataset.features(), dataset.target
    results: list[MatrixResult] = []
    for model_name in models:
        config = TrainingConfig(
            task="classification" if definition.task == "classification" else "regression",
            model=model_name,
            target_column=definition.target,
            group_column=definition.group_key,
            random_state=random_state,
        )
        model = build_model(config.task, model_name, X.iloc[split.train])
        model.fit(X.iloc[split.train], y.iloc[split.train])
        validation = evaluate(model, X.iloc[split.validation], y.iloc[split.validation], config.task)
        results.append(
            MatrixResult(
                dataset=definition.benchmark_id,
                benchmark_id=definition.benchmark_id,
                model=model_name,
                validation=validation,
                test=None,
                n_train=len(split.train),
                n_validation=len(split.validation),
                n_test=len(split.test),
            )
        )
    return results


def save_matrix(results: Iterable[MatrixResult], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(r) for r in results], indent=2, sort_keys=True), encoding="utf-8")
    return output
