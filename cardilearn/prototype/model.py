"""CardiLearn representation models.

Two capacity tiers are provided deliberately:

* ``CardiLearnProto`` is the small software-integration model used for fast
  tests and CPU development.
* ``CardiLearnLarge`` is the biologically serious research model. It expands
  transcriptome coverage, representation width, program capacity, and deep
  gene-program processing to a foundation-model-scale parameter regime.

Neither model is biologically validated merely by being large. Scientific
claims still require locked real-data training, held-out evaluation, and
independent validation.
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
    O(KG), avoiding an O(G^2) self-attention matrix over all genes.
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
    """CPU-feasible CardiLearn integration model."""

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


class CardiLearnLarge(nn.Module):
    """Large CardiLearn research architecture.

    Default scale targets roughly 100M+ trainable parameters once instantiated
    with a 20k-gene transcriptome. The architecture is intentionally designed
    around biological structure rather than simply widening an MLP:

    * ~20k-gene vocabulary instead of a 2k-gene prototype subset;
    * 1024-dimensional gene/program representations;
    * 32 learned molecular programs;
    * six deep Transformer blocks operating on program tokens;
    * 512-dimensional shared state and 128-dimensional private state;
    * context conditioning for species and assay;
    * full-transcriptome reconstruction head.

    Gene-level self-attention is avoided. The model first performs sparse,
    program-query cross-attention over the transcriptome, then spends depth
    modeling interactions among learned molecular programs. This makes scaling
    to ~20k genes substantially more tractable than dense O(G^2) attention.
    """

    def __init__(
        self,
        n_genes: int,
        n_species: int,
        n_assays: int,
        n_cell_types: int,
        gene_dim: int = 1024,
        n_programs: int = 32,
        n_layers: int = 6,
        n_heads: int = 16,
        ff_mult: int = 4,
        shared_dim: int = 512,
        private_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if gene_dim % n_heads != 0:
            raise ValueError("gene_dim must be divisible by n_heads")
        if n_genes < 1000:
            raise ValueError("CardiLearnLarge expects a substantial transcriptome; use CardiLearnProto for toy-scale tests")

        self.n_genes = n_genes
        self.gene_dim = gene_dim
        self.n_programs = n_programs

        self.context = ContextEncoder(n_species, n_assays, hidden_dim=256)
        self.gene_embedding = nn.Parameter(torch.randn(n_genes, gene_dim) * 0.02)
        self.expression_projection = nn.Linear(1, gene_dim)
        self.program_queries = nn.Parameter(torch.randn(n_programs, gene_dim) * 0.02)

        self.gene_to_program = nn.MultiheadAttention(
            gene_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(gene_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(gene_dim, ff_mult * gene_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * gene_dim, gene_dim),
        )
        self.cross_norm2 = nn.LayerNorm(gene_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=gene_dim,
            nhead=n_heads,
            dim_feedforward=ff_mult * gene_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.program_transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.context_gamma = nn.Linear(256, gene_dim)
        self.context_beta = nn.Linear(256, gene_dim)
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
            nn.Linear(gene_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, private_dim),
        )

        self.maturity = nn.Sequential(
            nn.Linear(shared_dim, 256), nn.GELU(), nn.Linear(256, 1)
        )
        self.injury = nn.Sequential(
            nn.Linear(shared_dim, 256), nn.GELU(), nn.Linear(256, 1)
        )
        self.cell_type = nn.Sequential(
            nn.Linear(shared_dim, 256), nn.GELU(), nn.Linear(256, n_cell_types)
        )

        # Transcriptome-scale decoder. This is deliberately explicit rather
        # than a tiny bottleneck decoder: reconstructing the held-out gene
        # space forces the latent state to retain broad molecular information.
        self.decoder = nn.Sequential(
            nn.Linear(shared_dim + private_dim + 256, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, n_genes),
        )

    def encode(self, x: torch.Tensor, species: torch.Tensor, assay: torch.Tensor):
        if x.ndim != 2 or x.shape[1] != self.n_genes:
            raise ValueError(f"Expected x with shape [batch, {self.n_genes}], got {tuple(x.shape)}")

        context = self.context(species, assay)
        gene_tokens = self.expression_projection(x.unsqueeze(-1))
        gene_tokens = gene_tokens + self.gene_embedding.unsqueeze(0)
        queries = self.program_queries.unsqueeze(0).expand(x.shape[0], -1, -1)

        programs, attention = self.gene_to_program(
            queries,
            gene_tokens,
            gene_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        programs = self.cross_norm(programs + queries)
        programs = self.cross_norm2(programs + self.cross_ffn(programs))

        gamma = 1.0 + self.context_gamma(context).unsqueeze(1)
        beta = self.context_beta(context).unsqueeze(1)
        programs = gamma * programs + beta
        programs = self.program_transformer(programs)

        pool_logits = self.program_pool(programs).squeeze(-1)
        pool_weights = torch.softmax(pool_logits, dim=-1).unsqueeze(-1)
        molecular = torch.sum(pool_weights * programs, dim=1)

        z_shared = self.shared_encoder(molecular)
        z_private = self.private_encoder(molecular)
        return z_shared, z_private, context, programs, attention

    def forward(self, x: torch.Tensor, species: torch.Tensor, assay: torch.Tensor) -> CardiLearnOutput:
        z_shared, z_private, context, programs, attention = self.encode(x, species, assay)
        reconstruction = self.decoder(torch.cat([z_shared, z_private, context], dim=-1))
        return CardiLearnOutput(
            z_shared=z_shared,
            z_private=z_private,
            maturity=self.maturity(z_shared).squeeze(-1),
            injury=self.injury(z_shared).squeeze(-1),
            cell_type=self.cell_type(z_shared),
            reconstruction=reconstruction,
            program_tokens=programs,
            program_attention=attention,
        )
