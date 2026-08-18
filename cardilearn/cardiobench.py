"""CardiBench-compatible benchmark definitions and leakage-safe split policies.

The loader accepts JSON or YAML benchmark definitions matching the CardiBench
BenchmarkDefinition contract. No benchmark data are redistributed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .splitting import SplitIndices


_ALLOWED_TASKS = {
    "classification", "regression", "retrieval", "representation",
    "detection", "characterization", "generalization",
}


@dataclass(frozen=True)
class BenchmarkDefinition:
    benchmark_id: str
    version: str
    task: str
    target: str
    split_policy: str
    group_key: str
    modalities: tuple[str, ...] = ()
    phenotype_ontology: str | None = None
    test_labels_private: bool = False
    source_datasets: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, obj: dict[str, Any]) -> "BenchmarkDefinition":
        required = {"benchmark_id", "version", "task", "target", "split_policy", "group_key"}
        missing = required - set(obj)
        if missing:
            raise ValueError(f"benchmark definition missing fields: {sorted(missing)}")
        task = str(obj["task"])
        if task not in _ALLOWED_TASKS:
            raise ValueError(f"unsupported benchmark task: {task}")
        return cls(
            benchmark_id=str(obj["benchmark_id"]),
            version=str(obj["version"]),
            task=task,
            target=str(obj["target"]),
            split_policy=str(obj["split_policy"]),
            group_key=str(obj["group_key"]),
            modalities=tuple(obj.get("modalities", ())),
            phenotype_ontology=obj.get("phenotype_ontology"),
            test_labels_private=bool(obj.get("test_labels_private", False)),
            source_datasets=tuple(obj.get("source_datasets", ())),
            metrics=tuple(obj.get("metrics", ())),
        )


def load_definition(path: str | Path) -> BenchmarkDefinition:
    """Load a benchmark definition from JSON or YAML."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML benchmark loading requires the 'yaml' extra") from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError("benchmark definition must be .json, .yaml, or .yml")
    if not isinstance(payload, dict):
        raise ValueError("benchmark definition must decode to an object")
    return BenchmarkDefinition.from_mapping(payload)


def validate_definition_against_frame(definition: BenchmarkDefinition, frame: pd.DataFrame) -> None:
    """Fail fast when a benchmark definition cannot be applied to a dataset table."""
    missing = [c for c in (definition.target, definition.group_key) if c not in frame.columns]
    if missing:
        raise ValueError(f"benchmark {definition.benchmark_id} requires missing columns: {missing}")
    if frame[definition.target].isna().any():
        raise ValueError("benchmark target contains missing values")
    if frame[definition.group_key].isna().any():
        raise ValueError("benchmark grouping column contains missing values")


def holdout_by_column(
    frame: pd.DataFrame,
    column: str,
    held_out_value: Any,
    *,
    validation_fraction: float = 0.20,
    random_state: int = 42,
) -> SplitIndices:
    """Create an external-style test set from one metadata level.

    The held-out level is never used for model selection. The remaining rows are
    partitioned into train/validation using deterministic group-aware splitting.
    """
    from .config import SplitConfig
    from .splitting import split_frame

    if column not in frame.columns:
        raise KeyError(f"holdout column not found: {column}")
    test = frame.index[frame[column] == held_out_value].to_numpy()
    dev_mask = frame[column] != held_out_value
    dev_positions = frame.index[dev_mask].to_numpy()
    if len(test) == 0:
        raise ValueError(f"no rows found for held-out value {held_out_value!r}")
    if len(dev_positions) < 3:
        raise ValueError("not enough development rows after holdout")

    dev = frame.loc[dev_positions].reset_index(drop=True)
    if "group_id" in dev.columns:
        groups = dev["group_id"]
    elif "sample_id" in dev.columns:
        groups = dev["sample_id"]
    else:
        groups = None
    cfg = SplitConfig(
        test_size=validation_fraction,
        validation_size=validation_fraction,
        random_state=random_state,
        stratify=False,
    )
    local = split_frame(dev, cfg, target=dev.iloc[:, 0], groups=groups)
    # split_frame's test becomes the development validation partition here.
    train_local = dev_positions[local.train]
    val_local = dev_positions[local.test]
    return SplitIndices(train=train_local, validation=val_local, test=test)
