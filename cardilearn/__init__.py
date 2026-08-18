"""Virelion CardiLearn: reproducible ML for real cardiac datasets."""

__version__ = "0.2.0"

from .dataset_card import DatasetCard
from .registry import ModelRegistry
from .schema import DatasetSpec, FeatureManifest
from .validation import IntegrityReport, validate_dataset

__all__ = ["DatasetCard", "DatasetSpec", "FeatureManifest", "IntegrityReport", "ModelRegistry", "validate_dataset"]
