"""Virelion CardiLearn: reproducible ML for real cardiac datasets."""

__version__ = "0.4.0"

from .benchmark_protocol import BenchmarkSpec, compare_seeded_scores, rank_models, summarize_repeated_scores
from .config_loader import ConfigError, load_yaml_config, validate_reproducibility_config
from .dataset_card import DatasetCard
from .fusion import align_modalities, concatenate_embeddings
from .modalities import OmicsMatrix, Waveform
from .multimodal import CardiLearnX, CardiacFusionCore, SignalPatchEncoder, cosine_alignment_loss, modality_dropout
from .registry import ModelRegistry
from .reproducibility import (
    ReproducibilityManifest,
    config_fingerprint,
    dataframe_fingerprint,
    fingerprint_ids,
    fingerprint_mapping,
    load_manifest,
    make_manifest,
    save_manifest,
)
from .schema import DatasetSpec, FeatureManifest
from .validation import IntegrityReport, validate_dataset

__all__ = [
    "BenchmarkSpec",
    "CardiacFusionCore",
    "CardiLearnX",
    "ConfigError",
    "DatasetCard",
    "DatasetSpec",
    "FeatureManifest",
    "IntegrityReport",
    "ModelRegistry",
    "OmicsMatrix",
    "ReproducibilityManifest",
    "SignalPatchEncoder",
    "Waveform",
    "align_modalities",
    "compare_seeded_scores",
    "config_fingerprint",
    "concatenate_embeddings",
    "cosine_alignment_loss",
    "dataframe_fingerprint",
    "fingerprint_ids",
    "fingerprint_mapping",
    "load_manifest",
    "load_yaml_config",
    "make_manifest",
    "modality_dropout",
    "rank_models",
    "save_manifest",
    "summarize_repeated_scores",
    "validate_dataset",
    "validate_reproducibility_config",
]
