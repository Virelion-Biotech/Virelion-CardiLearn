"""Multimodal cardiac latent-state architecture for CardiLearn-X.

The fusion model separates modality-private information from a shared cardiac
state. It supports transcriptomic, ECG, and MEA/field-potential embeddings,
missing-modality training, cross-modal alignment, phenotype heads, and
uncertainty-aware state quality estimation. Raw modality preprocessing remains
owned by the respective Virelion modality packages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MultimodalState:
    """Outputs of CardiLearn-X fusion."""
    z_shared: torch.Tensor
    private: dict[str, torch.Tensor]
    modality_quality: dict[str, torch.Tensor]
    maturation: torch.Tensor
    phenotype: torch.Tensor
    injury: torch.Tensor
    coupling: torch.Tensor
    uncertainty: torch.Tensor
    availability: torch.Tensor


class SignalPatchEncoder(nn.Module):
    """Generic 1-D signal encoder for ECG/MEA tensors.

    Input is [batch, channels, time]. Patchifying before attention keeps the
    sequence length explicit and lets upstream packages provide modality-
    specific channel/normalization conventions.
    """

    def __init__(self, in_channels: int, d_model: int = 512, patch_size: int = 64,
                 depth: int = 6, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.patch_size = patch_size
        self.projection = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3:
            raise ValueError("signal must have shape [batch, channels, time]")
        tokens = self.projection(signal).transpose(1, 2)
        cls = self.cls.expand(signal.shape[0], -1, -1)
        return self.norm(self.transformer(torch.cat([cls, tokens], dim=1))[:, 0])


class ModalityProjector(nn.Module):
    """Maps modality-specific encoder states into a common fusion space."""

    def __init__(self, input_dim: int, fusion_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QualityHead(nn.Module):
    """Estimates observation quality/reliability from a modality embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


class CardiacFusionCore(nn.Module):
    """Cross-modal attention core producing a shared cardiac state.

    Missing modalities are represented by a learned null token and an explicit
    availability mask. Modality identity embeddings prevent accidental
    equivalence of RNA, ECG, and MEA tokens while the shared state is trained
    by downstream tasks and cross-modal alignment objectives.
    """

    def __init__(self, fusion_dim: int = 768, depth: int = 8, heads: int = 12,
                 dropout: float = 0.1, modalities: tuple[str, ...] = ("rna", "ecg", "mea")) -> None:
        super().__init__()
        if fusion_dim % heads:
            raise ValueError("fusion_dim must be divisible by heads")
        self.modalities = modalities
        self.modality_embedding = nn.Parameter(torch.randn(len(modalities), fusion_dim) * 0.02)
        self.null_token = nn.Parameter(torch.zeros(1, fusion_dim))
        self.cls = nn.Parameter(torch.zeros(1, fusion_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=heads, dim_feedforward=4 * fusion_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(fusion_dim)
        self.quality = nn.ModuleDict({m: QualityHead(fusion_dim) for m in modalities})

    def forward(self, projected: Mapping[str, torch.Tensor], available: Mapping[str, torch.Tensor] | None = None):
        if not projected:
            raise ValueError("at least one modality embedding is required")
        batch = next(iter(projected.values())).shape[0]
        device = next(iter(projected.values())).device
        tokens, availability, quality = [], [], {}
        for i, modality in enumerate(self.modalities):
            x = projected.get(modality)
            if x is None:
                x = self.null_token.expand(batch, -1)
                present = torch.zeros(batch, dtype=torch.bool, device=device)
            else:
                if x.ndim != 2:
                    raise ValueError(f"{modality} embedding must be [batch, dim]")
                present = torch.ones(batch, dtype=torch.bool, device=x.device)
                if available is not None and modality in available:
                    present = available[modality].bool()
                x = torch.where(present.unsqueeze(-1), x, self.null_token.expand(batch, -1))
            token = x + self.modality_embedding[i].unsqueeze(0)
            tokens.append(token)
            availability.append(present)
            quality[modality] = self.quality[modality](token)
        sequence = torch.stack(tokens, dim=1)
        cls = self.cls.expand(batch, -1).unsqueeze(1)
        fused = self.norm(self.transformer(torch.cat([cls, sequence], dim=1))[:, 0])
        return fused, quality, torch.stack(availability, dim=1)


class CardiLearnX(nn.Module):
    """Unified multimodal cardiac latent-state model.

    ``encoders`` are expected to return modality embeddings. This deliberate
    boundary lets CardiLearn, ElectroTrace, and CardioScore evolve independently
    while sharing a rigorously defined fusion representation.
    """

    def __init__(self, encoder_dims: Mapping[str, int], fusion_dim: int = 768,
                 private_dim: int = 256, n_phenotypes: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        required = {"rna", "ecg", "mea"}
        if set(encoder_dims) != required:
            raise ValueError("encoder_dims must contain exactly rna, ecg, and mea")
        self.modalities = tuple(sorted(required))
        self.projectors = nn.ModuleDict({m: ModalityProjector(encoder_dims[m], fusion_dim) for m in self.modalities})
        self.private = nn.ModuleDict({m: nn.Sequential(nn.Linear(fusion_dim, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, private_dim)) for m in self.modalities})
        self.fusion = CardiacFusionCore(fusion_dim=fusion_dim, modalities=self.modalities, dropout=dropout)
        self.maturation = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.phenotype = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, n_phenotypes))
        self.injury = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.coupling = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.uncertainty = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, embeddings: Mapping[str, torch.Tensor], available: Mapping[str, torch.Tensor] | None = None) -> MultimodalState:
        projected = {m: self.projectors[m](embeddings[m]) for m in embeddings if m in self.projectors}
        z_shared, quality, availability = self.fusion(projected, available=available)
        private = {m: self.private[m](projected[m]) for m in projected}
        return MultimodalState(
            z_shared=z_shared,
            private=private,
            modality_quality=quality,
            maturation=self.maturation(z_shared).squeeze(-1),
            phenotype=self.phenotype(z_shared),
            injury=self.injury(z_shared).squeeze(-1),
            coupling=self.coupling(z_shared).squeeze(-1),
            uncertainty=F.softplus(self.uncertainty(z_shared).squeeze(-1)),
            availability=availability,
        )

    def encode(self, embeddings: Mapping[str, torch.Tensor], available: Mapping[str, torch.Tensor] | None = None) -> torch.Tensor:
        return self.forward(embeddings, available=available).z_shared


def cosine_alignment_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE for genuinely matched cross-modal observations."""
    if z_a.shape != z_b.shape:
        raise ValueError("paired embeddings must have identical shape")
    a = F.normalize(z_a, dim=-1)
    b = F.normalize(z_b, dim=-1)
    logits = (a @ b.T) / temperature
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def modality_dropout(embeddings: Mapping[str, torch.Tensor], probability: float = 0.2) -> dict[str, torch.Tensor]:
    """Randomly drop complete modalities during training; never alters values."""
    if not 0.0 <= probability < 1.0:
        raise ValueError("probability must be in [0, 1)")
    out = dict(embeddings)
    for modality, value in embeddings.items():
        if torch.rand((), device=value.device) < probability:
            out.pop(modality)
    if not out:
        modality = next(iter(embeddings))
        out[modality] = embeddings[modality]
    return out
