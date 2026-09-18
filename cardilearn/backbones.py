"""Standardized optional adapters for external transcriptomic encoders."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Protocol


class EncoderAdapter(Protocol):
    name: str

    def encode(self, data: Any) -> Any:
        """Return one representation per biological observation."""


@dataclass(frozen=True)
class CallableEncoderAdapter:
    """Adapter for any callable encoder with an explicit provenance label."""

    name: str
    encoder: Callable[[Any], Any]
    provenance: dict[str, str]

    def encode(self, data: Any) -> Any:
        return self.encoder(data)


@dataclass(frozen=True)
class ObjectEncoderAdapter:
    """Adapter around objects exposing a conventional encode/transform method."""

    name: str
    model: Any
    method: str = "encode"

    def encode(self, data: Any) -> Any:
        fn = getattr(self.model, self.method, None)
        if fn is None or not callable(fn):
            raise TypeError(f"{self.name} does not expose callable '{self.method}'")
        return fn(data)


@dataclass(frozen=True)
class ExternalEncoderSpec:
    """Declarative external-model entry used by the benchmark matrix."""

    name: str
    package: str
    source_repo: str
    method: str
    notes: str


EXTERNAL_ENCODERS = (
    ExternalEncoderSpec(
        name="geneformer",
        package="geneformer",
        source_repo="https://github.com/jkobject/geneformer",
        method="encode",
        notes="External pretrained rank-based transcriptomic representation; adapter uses the installed API supplied by the caller.",
    ),
    ExternalEncoderSpec(
        name="scgpt",
        package="scgpt",
        source_repo="https://github.com/bowang-lab/scGPT",
        method="encode",
        notes="External gene/value Transformer representation; adapter boundary intentionally avoids importing optional dependencies.",
    ),
    ExternalEncoderSpec(
        name="uce",
        package="uce",
        source_repo="https://github.com/snap-stanford/UCE",
        method="encode",
        notes="External zero-shot cell representation; caller supplies the repository-specific model wrapper.",
    ),
    ExternalEncoderSpec(
        name="scvi",
        package="scvi",
        source_repo="https://github.com/scverse/scvi-tools",
        method="get_latent_representation",
        notes="Probabilistic latent representation; callable is resolved only when the optional dependency is installed.",
    ),
)


def import_optional_model(module_name: str, object_path: str) -> Any:
    """Resolve an optional model object without making it a core dependency."""
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_path.split("."):
        obj = getattr(obj, part)
    return obj


def wrap_object(model: Any, *, name: str, method: str = "encode") -> ObjectEncoderAdapter:
    if not name.strip():
        raise ValueError("adapter name must be non-empty")
    return ObjectEncoderAdapter(name=name, model=model, method=method)


def known_external_encoder_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in EXTERNAL_ENCODERS)
