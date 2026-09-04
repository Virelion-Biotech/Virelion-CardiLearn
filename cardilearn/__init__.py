"""Virelion CardiLearn: reproducible ML for real cardiac datasets."""

__version__ = "0.3.0"

from .benchmark_protocol import BenchmarkSpec, compare_seeded_scores, rank_models, summarize_repeated_scores
from .dataset_card import DatasetCard
from .registry import ModelRegistry
from .schema import DatasetSpec, FeatureManifest
from .validation import IntegrityReport, validate_dataset

__all__ = [
    "BenchmarkSpec",
    "DatasetCard",
    "DatasetSpec",
    "FeatureManifest",
    "IntegrityReport",
    "ModelRegistry",
    "compare_seeded_scores",
    "rank_models",
    "summarize_repeated_scores",
    "validate_dataset",
]
