"""CardiLearn research representation model.

This module contains the experimental, modular transcriptomic representation path.
It deliberately keeps biological context out of the shared encoder unless a caller
explicitly chooses otherwise. The default architecture uses:
- separate gene-identity and expression-value representations;
- memory-bounded gene-to-program routing with optional GRN priors;
- a short program-token backbone (Transformer or optional Mamba);
- shared/private latent states;
- a factorized negative-binomial decoder rather than a dense gene-output MLP.

The implementation is an empirical model, not a scientific validity claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .conserved import ConservedGeneIdentity


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.strength * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, strength: float = 0.0) -> None:
        super().__init__()
        if strength < 0:
            raise ValueError("strength must be non-negative")
        self.strength = float(strength)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.strength)


def fractional_rank(values: torch.Tensor) -> torch.Tensor:
    """Return per-row fractional ranks in [0, 1], with zeros retaining zero.

    Ties are resolved deterministically by the original column order. This is
    provided as an explicit alternative input representation, not the default.
    """
    if values.ndim != 2:
        raise ValueError("values must have shape [batch, genes]")
    order = torch.argsort(values, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    positions = torch.arange(values.shape[1], device=values.device, dtype=torch.float32)
    positions = positions.unsqueeze(0).expand(values.shape[0], -1)
    ranks.scatter_(1, order, positions)
    denom = max(values.shape[1] - 1, 1)
    ranks = 1.0 - ranks / denom
    return torch.where(values > 0, ranks, torch.zeros_like(ranks))


class GeneValueEncoder(nn.Module):
    """Combine trainable gene identity with a selectable expression encoding."""

    def __init__(
        self,
        n_genes: int,
        dim: int,
        *,
        value_style: str = "continuous",
        dropout: float = 0.1,
        conserved_group_ids: torch.Tensor | None = None,
        n_conserved_groups: int | None = None,
    ) -> None:
        super().__init__()
        if n_genes < 1 or dim < 1:
            raise ValueError("n_genes and dim must be positive")
        if value_style not in {"continuous", "rank"}:
            raise ValueError("value_style must be 'continuous' or 'rank'")
        self.n_genes = n_genes
        self.dim = dim
        self.value_style = value_style
        self.gene_embedding = nn.Parameter(torch.empty(n_genes, dim))
        self.mask_embedding = nn.Parameter(torch.empty(1, 1, dim))
        self.conserved_identity = None
        if conserved_group_ids is not None:
            if n_conserved_groups is None:
                raise ValueError("n_conserved_groups is required with conserved_group_ids")
            self.conserved_identity = ConservedGeneIdentity(
                n_genes, dim, conserved_group_ids, n_conserved_groups
            )
        elif n_conserved_groups is not None:
            raise ValueError("conserved_group_ids is required with n_conserved_groups")
        nn.init.normal_(self.gene_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)
        self.value_projection = nn.Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, counts: torch.Tensor, masked: torch.Tensor | None = None) -> torch.Tensor:
        if counts.ndim != 2 or counts.shape[1] != self.n_genes:
            raise ValueError(f"expected [batch, {self.n_genes}] counts, got {tuple(counts.shape)}")
        if not torch.is_floating_point(counts):
            counts = counts.float()
        if torch.any(~torch.isfinite(counts)) or torch.any(counts < 0):
            raise ValueError("counts must be finite and non-negative")
        if self.value_style == "continuous":
            values = torch.log1p(counts)
        else:
            values = fractional_rank(counts)
        identity = self.gene_embedding
        if self.conserved_identity is not None:
            identity = identity + self.conserved_identity()
        tokens = identity.unsqueeze(0) + self.value_projection(values.unsqueeze(-1))
        if masked is not None:
            if masked.shape != counts.shape:
                raise ValueError("masked must have the same shape as counts")
            tokens = tokens + masked.to(tokens.dtype).unsqueeze(-1) * self.mask_embedding
        return tokens


class GRNProgramRouter(nn.Module):
    """Route genes into learned molecular programs without a dense G×G attention map.

    A trainable gene→program assignment is combined with optional externally
    derived GRN/pathway prior logits. The softmax is evaluated in gene chunks so
    peak memory does not scale with the complete [batch, genes, programs] tensor.
    """

    def __init__(
        self,
        n_genes: int,
        n_programs: int,
        dim: int,
        *,
        chunk_size: int = 2048,
        prior: torch.Tensor | None = None,
        prior_strength: float = 0.0,
    ) -> None:
        super().__init__()
        if n_genes < 1 or n_programs < 1 or dim < 1:
            raise ValueError("n_genes, n_programs and dim must be positive")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if prior_strength < 0:
            raise ValueError("prior_strength must be non-negative")
        self.n_genes = n_genes
        self.n_programs = n_programs
        self.dim = dim
        self.chunk_size = chunk_size
        self.prior_strength = float(prior_strength)
        self.assignment_logits = nn.Parameter(torch.empty(n_genes, n_programs))
        nn.init.normal_(self.assignment_logits, mean=0.0, std=0.02)
        if prior is None:
            prior_tensor = torch.zeros(n_genes, n_programs, dtype=torch.float32)
        else:
            prior_tensor = torch.as_tensor(prior, dtype=torch.float32)
            if tuple(prior_tensor.shape) != (n_genes, n_programs):
                raise ValueError(
                    f"prior must have shape {(n_genes, n_programs)}, got {tuple(prior_tensor.shape)}"
                )
            if not torch.isfinite(prior_tensor).all():
                raise ValueError("prior contains non-finite values")
        self.register_buffer("prior", prior_tensor, persistent=False)
        self.gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def _logits_chunk(self, gate: torch.Tensor, start: int, end: int) -> torch.Tensor:
        assignment = self.assignment_logits[start:end].T.unsqueeze(0).to(dtype=gate.dtype, device=gate.device)
        prior = self.prior[start:end].T.unsqueeze(0).to(dtype=gate.dtype, device=gate.device)
        return assignment + self.prior_strength * prior + gate[:, None, start:end]

    def forward(self, gene_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if gene_tokens.ndim != 3 or gene_tokens.shape[1:] != (self.n_genes, self.dim):
            raise ValueError(
                f"gene_tokens must have shape [batch, {self.n_genes}, {self.dim}], "
                f"got {tuple(gene_tokens.shape)}"
            )
        gate = self.gate(gene_tokens).squeeze(-1)
        batch = gene_tokens.shape[0]
        device = gene_tokens.device
        max_logits = torch.full(
            (batch, self.n_programs),
            -torch.inf,
            dtype=gene_tokens.dtype,
            device=device,
        )
        for start in range(0, self.n_genes, self.chunk_size):
            end = min(start + self.chunk_size, self.n_genes)
            max_logits = torch.maximum(max_logits, self._logits_chunk(gate, start, end).amax(dim=-1))

        accum_dtype = torch.float32 if gene_tokens.dtype in {torch.float16, torch.bfloat16} else gene_tokens.dtype
        denom = torch.zeros(batch, self.n_programs, dtype=accum_dtype, device=device)
        for start in range(0, self.n_genes, self.chunk_size):
            end = min(start + self.chunk_size, self.n_genes)
            weights = torch.exp(self._logits_chunk(gate, start, end) - max_logits.unsqueeze(-1)).to(accum_dtype)
            denom += weights.sum(dim=-1)

        programs = torch.zeros(
            batch, self.n_programs, self.dim, dtype=accum_dtype, device=device
        )
        for start in range(0, self.n_genes, self.chunk_size):
            end = min(start + self.chunk_size, self.n_genes)
            weights = torch.exp(self._logits_chunk(gate, start, end) - max_logits.unsqueeze(-1)).to(accum_dtype)
            programs += torch.einsum("bkg,bgd->bkd", weights, gene_tokens.to(accum_dtype))
        programs = programs / denom.clamp_min(torch.finfo(programs.dtype).tiny).unsqueeze(-1)
        program_mass = denom / denom.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(denom.dtype).tiny)
        return programs.to(gene_tokens.dtype), program_mass.to(gene_tokens.dtype)

    @torch.no_grad()
    def top_genes(
        self,
        gene_tokens: torch.Tensor,
        *,
        k: int = 20,
        sample_index: int = 0,
    ) -> dict[int, list[tuple[int, float]]]:
        if not 0 <= sample_index < gene_tokens.shape[0]:
            raise IndexError("sample_index outside batch")
        if k < 1:
            raise ValueError("k must be positive")
        gate = self.gate(gene_tokens[sample_index : sample_index + 1]).squeeze(-1)[0]
        scores = self.assignment_logits.T + self.prior_strength * self.prior.T
        scores = scores + gate.unsqueeze(0)
        k = min(k, self.n_genes)
        values, indices = torch.topk(scores, k=k, dim=-1)
        return {
            program: [(int(idx), float(value)) for idx, value in zip(indices[program], values[program])]
            for program in range(self.n_programs)
        }


class ProgramBackbone(nn.Module):
    """Program-token backbone with an optional lazy Mamba implementation."""

    def __init__(
        self,
        dim: int,
        *,
        layers: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
        backend: str = "transformer",
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if layers < 1:
            raise ValueError("layers must be positive")
        if backend not in {"transformer", "mamba"}:
            raise ValueError("backend must be 'transformer' or 'mamba'")
        self.backend = backend
        if backend == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=4 * dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=layers)
        else:
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise ImportError(
                    "Mamba backend requires the optional 'mamba-ssm' package"
                ) from exc
            self.backbone = nn.Sequential(*[Mamba(d_model=dim, d_state=16, d_conv=4, expand=2) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, programs: torch.Tensor) -> torch.Tensor:
        if programs.ndim != 3:
            raise ValueError("programs must have shape [batch, programs, dim]")
        return self.norm(self.backbone(programs))


class FactorizedNBDecoder(nn.Module):
    """Gene-factorized negative-binomial decoder.

    The decoder represents each gene with an output embedding and scores it
    against a cell state. This avoids a wide dense hidden→gene matrix and makes
    decoder capacity scale approximately linearly in embedding width × gene count.
    """

    def __init__(
        self,
        n_genes: int,
        cell_dim: int,
        *,
        decoder_dim: int = 256,
    ) -> None:
        super().__init__()
        if n_genes < 1 or cell_dim < 1 or decoder_dim < 1:
            raise ValueError("dimensions must be positive")
        self.n_genes = n_genes
        self.decoder_dim = decoder_dim
        self.cell_projection = nn.Sequential(
            nn.Linear(cell_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.GELU(),
        )
        self.gene_projection = nn.Parameter(torch.empty(n_genes, decoder_dim))
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        nn.init.normal_(self.gene_projection, mean=0.0, std=0.02)

    def forward(self, cell_state: torch.Tensor, library_size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if cell_state.ndim != 2:
            raise ValueError("cell_state must be [batch, dim]")
        if library_size.ndim != 1 or library_size.shape[0] != cell_state.shape[0]:
            raise ValueError("library_size must be [batch]")
        if torch.any(~torch.isfinite(library_size)) or torch.any(library_size <= 0):
            raise ValueError("library_size must be finite and positive")
        projected = self.cell_projection(cell_state)
        log_mu = (
            projected @ self.gene_projection.T
            + self.gene_bias.unsqueeze(0)
            + torch.log(library_size).unsqueeze(-1)
        )
        log_mu = torch.clamp(log_mu, min=-20.0, max=20.0)
        mu = torch.exp(log_mu)
        theta = F.softplus(self.log_theta).unsqueeze(0) + 1e-4
        return mu, theta


@dataclass
class ResearchOutput:
    z_program: torch.Tensor
    z_shared: torch.Tensor
    z_private: torch.Tensor
    reconstruction_mu: torch.Tensor
    reconstruction_theta: torch.Tensor
    masked_prediction: torch.Tensor
    maturation: torch.Tensor
    injury: torch.Tensor
    cell_type: torch.Tensor
    species_logits: torch.Tensor | None
    program_mass: torch.Tensor


class CardiLearnResearch(nn.Module):
    """Cardiac transcriptomic representation model with modular biological routing."""

    def __init__(
        self,
        n_genes: int,
        n_species: int,
        n_assays: int,
        n_cell_types: int,
        *,
        n_tissues: int = 1,
        gene_dim: int = 256,
        n_programs: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        shared_dim: int = 384,
        private_dim: int = 128,
        decoder_dim: int = 256,
        dropout: float = 0.1,
        value_style: str = "continuous",
        backbone: str = "transformer",
        grn_prior: torch.Tensor | None = None,
        grn_prior_strength: float = 0.0,
        router_chunk_size: int = 2048,
        species_adversarial_strength: float = 0.0,
    ) -> None:
        super().__init__()
        if n_genes < 1:
            raise ValueError("n_genes must be positive")
        for name, value in (("n_species", n_species), ("n_assays", n_assays), ("n_cell_types", n_cell_types), ("n_tissues", n_tissues)):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        self.n_genes = n_genes
        self.gene_dim = gene_dim
        self.n_programs = n_programs
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.value_style = value_style

        self.input_encoder = GeneValueEncoder(n_genes, gene_dim, value_style=value_style, dropout=dropout)
        self.router = GRNProgramRouter(
            n_genes,
            n_programs,
            gene_dim,
            chunk_size=router_chunk_size,
            prior=grn_prior,
            prior_strength=grn_prior_strength,
        )
        self.program_backbone = ProgramBackbone(
            gene_dim,
            layers=n_layers,
            heads=n_heads,
            dropout=dropout,
            backend=backbone,
        )
        self.program_pool = nn.Sequential(
            nn.LayerNorm(gene_dim),
            nn.Linear(gene_dim, 1),
        )
        self.shared_encoder = nn.Sequential(
            nn.Linear(gene_dim, gene_dim),
            nn.LayerNorm(gene_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gene_dim, shared_dim),
            nn.LayerNorm(shared_dim),
        )
        self.private_encoder = nn.Sequential(
            nn.Linear(gene_dim, max(gene_dim // 2, private_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(gene_dim // 2, private_dim), private_dim),
        )

        self.species_embedding = nn.Embedding(n_species, 32)
        self.assay_embedding = nn.Embedding(n_assays, 16)
        self.tissue_embedding = nn.Embedding(n_tissues, 16)
        context_dim = 64
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        decoder_context_dim = shared_dim + private_dim + 128
        self.decoder = FactorizedNBDecoder(
            n_genes,
            decoder_context_dim,
            decoder_dim=decoder_dim,
        )
        self.masked_cell_projection = nn.Sequential(
            nn.Linear(decoder_context_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.GELU(),
        )
        self.maturation = nn.Sequential(nn.Linear(shared_dim, 192), nn.GELU(), nn.Linear(192, 1))
        self.injury = nn.Sequential(nn.Linear(shared_dim, 192), nn.GELU(), nn.Linear(192, 1))
        self.cell_type = nn.Sequential(nn.Linear(shared_dim, 192), nn.GELU(), nn.Linear(192, n_cell_types))

        self.species_adversarial_strength = float(species_adversarial_strength)
        self.species_adversary = (
            nn.Sequential(nn.Linear(shared_dim, 192), nn.GELU(), nn.Linear(192, n_species))
            if species_adversarial_strength > 0
            else None
        )
        self.grl = GradientReversal(species_adversarial_strength)

    def _context(self, species: torch.Tensor, assay: torch.Tensor, tissue: torch.Tensor | None) -> torch.Tensor:
        if tissue is None:
            tissue = torch.zeros_like(species)
        for name, values in (("species", species), ("assay", assay), ("tissue", tissue)):
            if values.ndim != 1:
                raise ValueError(f"{name} must have shape [batch]")
        encoded = torch.cat(
            [
                self.species_embedding(species),
                self.assay_embedding(assay),
                self.tissue_embedding(tissue),
            ],
            dim=-1,
        )
        return self.context_projection(encoded)

    def encode(
        self,
        counts: torch.Tensor,
        species: torch.Tensor,
        assay: torch.Tensor,
        tissue: torch.Tensor | None = None,
        gene_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if counts.shape[0] != species.shape[0] or counts.shape[0] != assay.shape[0]:
            raise ValueError("counts, species and assay batch dimensions must agree")
        gene_tokens = self.input_encoder(counts, masked=gene_mask)
        programs, program_mass = self.router(gene_tokens)
        programs = self.program_backbone(programs)
        pool_logits = self.program_pool(programs).squeeze(-1)
        pool_weights = torch.softmax(pool_logits, dim=-1)
        z_program = torch.sum(programs * pool_weights.unsqueeze(-1), dim=1)
        z_shared = self.shared_encoder(z_program)
        z_private = self.private_encoder(z_program)
        return z_program, z_shared, z_private, program_mass

    def forward(
        self,
        counts: torch.Tensor,
        species: torch.Tensor,
        assay: torch.Tensor,
        *,
        tissue: torch.Tensor | None = None,
        library_size: torch.Tensor | None = None,
        gene_mask: torch.Tensor | None = None,
    ) -> ResearchOutput:
        if library_size is None:
            library_size = counts.sum(dim=-1).clamp_min(1.0)
        z_program, z_shared, z_private, program_mass = self.encode(counts, species, assay, tissue, gene_mask)
        context = self._context(species, assay, tissue)
        cell_state = torch.cat([z_shared, z_private, context], dim=-1)
        mu, theta = self.decoder(cell_state, library_size)
        masked_hidden = self.masked_cell_projection(cell_state)
        masked_prediction = masked_hidden @ self.decoder.gene_projection.T + self.decoder.gene_bias.unsqueeze(0)
        species_logits = None
        if self.species_adversary is not None:
            species_logits = self.species_adversary(self.grl(z_shared))
        return ResearchOutput(
            z_program=z_program,
            z_shared=z_shared,
            z_private=z_private,
            reconstruction_mu=mu,
            reconstruction_theta=theta,
            masked_prediction=masked_prediction,
            maturation=self.maturation(z_shared).squeeze(-1),
            injury=self.injury(z_shared).squeeze(-1),
            cell_type=self.cell_type(z_shared),
            species_logits=species_logits,
            program_mass=program_mass,
        )

    def parameter_count(self, trainable_only: bool = True) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def encode_cells(
        self,
        counts: torch.Tensor,
        species: torch.Tensor,
        assay: torch.Tensor,
        *,
        tissue: torch.Tensor | None = None,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode a large matrix in bounded mini-batches without changing weights."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, counts.shape[0], batch_size):
                end = min(start + batch_size, counts.shape[0])
                _, z_shared, _, _ = self.encode(
                    counts[start:end],
                    species[start:end],
                    assay[start:end],
                    None if tissue is None else tissue[start:end],
                )
                chunks.append(z_shared.detach())
        return torch.cat(chunks, dim=0)
