"""CardiLearn v0.1 structured cardiac-state representation model.

The prototype is deliberately small enough for CPU/12-GB RAM experimentation,
while retaining the core architectural idea:

expression -> gene tokens -> learned molecular programs -> context modulation
-> shared/private latent state -> biological heads + reconstruction.

It is not a clinical model and does not encode an absolute regeneration
probability. Regeneration supervision will be added as a sample-level
relational objective in a later training stage.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CardiLearnOutput:
    z_shared: torch.Tensor
    z_private: torch.Tensor
    maturity: torch.Tensor
    injury: torch.Tensor
    cell_type: torch.Tensor | None
    reconstruction: torch.Tensor
    program_tokens: torch.Tensor
    program_attention: torch.Tensor


class ContextEncoder(nn.Module):
    def __init__(self, n_species: int, n_assays: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.species_embedding = nn.Embedding(n_species, 16)
        self.assay_embedding = nn.Embedding(n_assays, 8)
        self.net = nn.Sequential(
            nn.Linear(24, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, species: torch.Tensor, assay: torch.Tensor) -> torch.Tensor:
        s = self.species_embedding(species)
        a = self.assay_embedding(assay)
        return self.net(torch.cat([s, a], dim=-1))


class GeneProgramEncoder(nn.Module):
    """Compress G gene tokens into K learned molecular program tokens.

    Cross-attention is program-query -> gene-key/value, so complexity is
    O(KG), avoiding an O(G^2) self-attention matrix at prototype scale.
    """

    def __init__(self, n_genes: int, gene_dim: int = 64, n_programs: int = 16, n_heads: int = 4) -> None:
        super().__init__()
        if gene_dim % n_heads != 0:
            raise ValueError("gene_dim must be divisible by n_heads")
        self.n_genes = n_genes
        self.gene_embedding = nn.Parameter(torch.randn(n_genes, gene_dim) * 0.02)
        self.expression_projection = nn.Linear(1, gene_dim)
        self.program_queries = nn.Parameter(torch.randn(n_programs, gene_dim) * 0.02)
        self.attention = nn.MultiheadAttention(gene_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(gene_dim)
        self.norm2 = nn.LayerNorm(gene_dim)
        self.ffn = nn.Sequential(
            nn.Linear(gene_dim, 2 * gene_dim),
            nn.GELU(),
            nn.Linear(2 * gene_dim, gene_dim),
        )
        self.pool_score = nn.Linear(gene_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape [batch, genes], got {tuple(x.shape)}")
        if x.shape[1] != self.n_genes:
            raise ValueError(f"Expected {self.n_genes} genes, got {x.shape[1]}")

        b = x.shape[0]
        gene_tokens = self.expression_projection(x.unsqueeze(-1))
        gene_tokens = gene_tokens + self.gene_embedding.unsqueeze(0)

        queries = self.program_queries.unsqueeze(0).expand(b, -1, -1)
        programs, attention = self.attention(
            queries,
            gene_tokens,
            gene_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        programs = self.norm1(programs + queries)
        programs = self.norm2(programs + self.ffn(programs))

        scores = self.pool_score(programs).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = torch.sum(weights * programs, dim=1)
        return pooled, programs, attention


class CardiLearnProto(nn.Module):
    """CPU-feasible CardiLearn v0.1 model."""

    def __init__(
        self,
        n_genes: int,
        n_species: int,
        n_assays: int,
        n_cell_types: int,
        gene_dim: int = 64,
        n_programs: int = 16,
        shared_dim: int = 128,
        private_dim: int = 32,
    ) -> None:
        super().__init__()
        self.context = ContextEncoder(n_species, n_assays, hidden_dim=64)
        self.program_encoder = GeneProgramEncoder(
            n_genes=n_genes,
            gene_dim=gene_dim,
            n_programs=n_programs,
            n_heads=4,
        )

        self.gamma = nn.Linear(64, gene_dim)
        self.beta = nn.Linear(64, gene_dim)

        self.shared_encoder = nn.Sequential(
            nn.Linear(gene_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, shared_dim),
        )
        self.private_encoder = nn.Sequential(
            nn.Linear(gene_dim, 64),
            nn.GELU(),
            nn.Linear(64, private_dim),
        )

        self.maturity = nn.Sequential(
            nn.Linear(shared_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.injury = nn.Sequential(
            nn.Linear(shared_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.cell_type = nn.Sequential(
            nn.Linear(shared_dim, 64), nn.GELU(), nn.Linear(64, n_cell_types)
        )

        self.decoder = nn.Sequential(
            nn.Linear(shared_dim + private_dim + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, n_genes),
        )

    def encode(self, x: torch.Tensor, species: torch.Tensor, assay: torch.Tensor):
        context = self.context(species, assay)
        molecular, programs, attention = self.program_encoder(x)
        gamma = 1.0 + self.gamma(context)
        beta = self.beta(context)
        contextual = gamma * molecular + beta
        z_shared = self.shared_encoder(contextual)
        z_private = self.private_encoder(contextual)
        return z_shared, z_private, context, programs, attention

    def forward(self, x: torch.Tensor, species: torch.Tensor, assay: torch.Tensor) -> CardiLearnOutput:
        z_shared, z_private, context, programs, attention = self.encode(x, species, assay)
        return CardiLearnOutput(
            z_shared=z_shared,
            z_private=z_private,
            maturity=self.maturity(z_shared).squeeze(-1),
            injury=self.injury(z_shared).squeeze(-1),
            cell_type=self.cell_type(z_shared),
            reconstruction=self.decoder(torch.cat([z_shared, z_private, context], dim=-1)),
            program_tokens=programs,
            program_attention=attention,
        )
