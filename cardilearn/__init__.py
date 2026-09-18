"""Virelion CardiLearn: reproducible ML for cardiac datasets."""

__version__ = "0.4.1"

from .backbones import EXTERNAL_ENCODERS, CallableEncoderAdapter, EncoderAdapter, ExternalEncoderSpec, known_external_encoder_names, wrap_object
from .benchmark_protocol import BenchmarkSpec, compare_seeded_scores, rank_models, summarize_repeated_scores
from .config_loader import ConfigError, load_yaml_config, validate_reproducibility_config
from .dataset_card import DatasetCard
from .fusion import align_modalities, concatenate_embeddings
from .modalities import OmicsMatrix, Waveform
from .objectives import ObjectiveSchedule, ObjectiveStage, ObjectiveWeights, cosine_alignment_loss, masked_log1p_loss, negative_binomial_nll, vicreg_loss
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

try:  # pragma: no cover - depends on optional torch installation
    from .multimodal import CardiLearnX, CardiacFusionCore, SignalPatchEncoder, modality_dropout
except ImportError:
    CardiLearnX = None
    CardiacFusionCore = None
    SignalPatchEncoder = None
    modality_dropout = None

try:  # pragma: no cover - depends on optional torch installation
    from .torch_training import CategoryEncoder, TorchTrainConfig, fit_research_model, load_checkpoint, save_checkpoint
except ImportError:
    CategoryEncoder = None
    TorchTrainConfig = None
    fit_research_model = None
    load_checkpoint = None
    save_checkpoint = None

try:  # pragma: no cover - depends on optional torch installation
    from .research_model import CardiLearnResearch, FactorizedNBDecoder, GeneValueEncoder, GRNProgramRouter, ProgramBackbone
except ImportError:
    CardiLearnResearch = None
    FactorizedNBDecoder = None
    GeneValueEncoder = None
    GRNProgramRouter = None
    ProgramBackbone = None

__all__ = [
    "BenchmarkSpec",
    "CardiacFusionCore",
    "CardiLearnResearch",
    "CardiLearnX",
    "CallableEncoderAdapter",
    "ConfigError",
    "DatasetCard",
    "DatasetSpec",
    "EncoderAdapter",
    "EXTERNAL_ENCODERS",
    "ExternalEncoderSpec",
    "FactorizedNBDecoder",
    "FeatureManifest",
    "GRNProgramRouter",
    "GeneValueEncoder",
    "IntegrityReport",
    "ModelRegistry",
    "OmicsMatrix",
    "ObjectiveSchedule", "ObjectiveStage", "ObjectiveWeights",    "ProgramBackbone",
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
    "known_external_encoder_names",
    "load_manifest",
    "load_yaml_config",
    "make_manifest",
    "masked_log1p_loss", "negative_binomial_nll", "vicreg_loss",    "modality_dropout",
    "rank_models",
    "save_manifest",
    "summarize_repeated_scores",
    "validate_dataset",
    "validate_reproducibility_config",
    "wrap_object",
    "CategoryEncoder", "TorchTrainConfig", "fit_research_model", "load_checkpoint", "save_checkpoint",
]
