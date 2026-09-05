"""CardiLearn representation models.

The research model is designed around the statistical structure of single-cell
transcriptomics rather than treating expression as an ordinary dense vector.
It uses transcriptome-scale gene identities, learned molecular programs,
count-aware decoding, masked-gene prediction, explicit technical context, and
species-adversarial pressure on the shared biological state.

``CardiLearnProto`` remains a small integration model. ``CardiLearnLarge`` is
the research architecture intended for large-scale pretraining.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


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
    nb_mu: torch.Tensor | None = None
    nb_theta: torch.Tensor | None = None
    masked_prediction: torch.Tensor | None = None
    species_logits: torch.Tensor | None = None


class ContextEncoder(nn.Module):
    def __init__(self, n_species: int, n_assays: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.species_embedding = nn.Embedding(n_species, 16)
        self.assay_embedding = nn.Embedding(n_assays, 8)
        self.net = nn.Sequential(
            nn.Linear(24, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, species: torch.Tensor, assay: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.species_embedding(species), self.assay_embedding(assay)], dim=-1))


class GeneProgramEncoder(nn.Module):
    """Compress G gene tokens into K learned molecular program tokens."""

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
        self.ffn = nn.Sequential(nn.Linear(gene_dim, 2 * gene_dim), nn.GELU(), nn.Linear(2 * gene_dim, gene_dim))
        self.pool_score = nn.Linear(gene_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.n_genes:
            raise ValueError(f"Expected x with shape [batch, {self.n_genes}], got {tuple(x.shape)}")
        gene_tokens = self.expression_projection(x.unsqueeze(-1)) + self.gene_embedding.unsqueeze(0)
        queries = self.program_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        programs, attention = self.attention(queries, gene_tokens, gene_tokens, need_weights=True, average_attn_weights=False)
        programs = self.norm1(programs + queries)
        programs = self.norm2(programs + self.ffn(programs))
        weights = torch.softmax(self.pool_score(programs).squeeze(-1), dim=-1).unsqueeze(-1)
        return torch.sum(weights * programs, dim=1), programs, attention


class CardiLearnProto(nn.Module):
    """CPU-feasible legacy integration model."""

    def __init__(self, n_genes: int, n_species: int, n_assays: int, n_cell_types: int,
                 gene_dim: int = 64, n_programs: int = 16, shared_dim: int = 128, private_dim: int = 32) -> None:
        super().__init__()
        self.context = ContextEncoder(n_species, n_assays, hidden_dim=64)
        self.program_encoder = GeneProgramEncoder(n_genes, gene_dim, n_programs, n_heads=4)
        self.gamma = nn.Linear(64, gene_dim)
        self.beta = nn.Linear(64, gene_dim)
        self.shared_encoder = nn.Sequential(nn.Linear(gene_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, shared_dim))
        self.private_encoder = nn.Sequential(nn.Linear(gene_dim, 64), nn.GELU(), nn.Linear(64, private_dim))
        self.maturity = nn.Sequential(nn.Linear(shared_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.injury = nn.Sequential(nn.Linear(shared_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.cell_type = nn.Sequential(nn.Linear(shared_dim, 64), nn.GELU(), nn.Linear(64, n_cell_types))
        self.decoder = nn.Sequential(nn.Linear(shared_dim + private_dim + 64, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 512), nn.GELU(), nn.Linear(512, n_genes))

    def encode(self, x: torch.Tensor, species: torch.Tensor, assay: torch.Tensor):
        context = self.context(species, assay)
        molecular, programs, attention = self.program_encoder(x)
        contextual = (1.0 + self.gamma(context)) * molecular + self.beta(context)
        return self.shared_encoder(contextual), self.private_encoder(contextual), context, programs, attention

    def forward(self, x: torch.Tensor, species: torch.Tensor, assay: torch.Tensor) -> CardiLearnOutput:
        z_shared, z_private, context, programs, attention = self.encode(x, species, assay)
        return CardiLearnOutput(z_shared, z_private, self.maturity(z_shared).squeeze(-1), self.injury(z_shared).squeeze(-1), self.cell_type(z_shared), self.decoder(torch.cat([z_shared, z_private, context], dim=-1)), programs, attention)


class CardiLearnLarge(nn.Module):
    """Transcriptome-scale CardiLearn research architecture.

    Biological/statistical design:

    * raw UMI/read counts are represented after ``log1p`` input stabilization,
      while reconstruction is a negative-binomial likelihood with a learned
      gene-specific dispersion;
    * library size is explicitly supplied to the generative decoder rather than
      forcing the latent state to encode sequencing depth;
    * masked-gene prediction creates a genuine self-supervised objective;
    * learned program tokens model coordinated gene programs without quadratic
      attention over all ~20k genes;
    * the shared latent is intended to capture biology while private state can
      retain cell/assay-specific variation;
    * species is adversarially predicted from the shared state, discouraging
      species identity from becoming the representation itself when orthologs
      have been harmonized upstream;
    * cell type, maturation and injury are auxiliary biological tasks, not the
      definition of the latent space.

    This architecture is intentionally much larger than the prototype. With
    20k genes it is in the same broad parameter regime as contemporary
    single-cell foundation models, but model size alone is not evidence of
    biological validity.
    """

    def __init__(self, n_genes: int, n_species: int, n_assays: int, n_cell_types: int,
                 gene_dim: int = 1024, n_programs: int = 64, n_layers: int = 8,
                 n_heads: int = 16, ff_mult: int = 4, shared_dim: int = 768,
                 private_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        if gene_dim % n_heads != 0:
            raise ValueError("gene_dim must be divisible by n_heads")
        if n_genes < 5000:
            raise ValueError("CardiLearnLarge expects transcriptome-scale input; use CardiLearnProto for toy-scale tests")
        self.n_genes = n_genes
        self.gene_dim = gene_dim
        self.n_programs = n_programs

        # Gene identity is a first-class object. Expression magnitude is a
        # separate signal, preventing the model from confusing gene identity
        # with count magnitude.
        self.gene_embedding = nn.Parameter(torch.randn(n_genes, gene_dim) * 0.02)
        self.expression_projection = nn.Sequential(nn.Linear(1, gene_dim), nn.LayerNorm(gene_dim), nn.GELU())
        self.program_queries = nn.Parameter(torch.randn(n_programs, gene_dim) * 0.02)
        self.gene_to_program = nn.MultiheadAttention(gene_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(gene_dim)
        self.cross_ffn = nn.Sequential(nn.Linear(gene_dim, ff_mult * gene_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * gene_dim, gene_dim))
        self.cross_norm2 = nn.LayerNorm(gene_dim)

        layer = nn.TransformerEncoderLayer(d_model=gene_dim, nhead=n_heads, dim_feedforward=ff_mult * gene_dim,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.program_transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.program_norm = nn.LayerNorm(gene_dim)
        self.program_pool = nn.Linear(gene_dim, 1)

        # Technical context is decoder-side by default. This prevents species
        # and assay labels from trivially defining the shared biological state.
        self.species_embedding = nn.Embedding(n_species, 64)
        self.assay_embedding = nn.Embedding(n_assays, 32)
        self.context_projection = nn.Sequential(nn.Linear(96, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 256))

        self.shared_encoder = nn.Sequential(nn.Linear(gene_dim, gene_dim), nn.LayerNorm(gene_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(gene_dim, shared_dim), nn.LayerNorm(shared_dim))
        self.private_encoder = nn.Sequential(nn.Linear(gene_dim, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, private_dim))

        self.maturity = nn.Sequential(nn.Linear(shared_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.injury = nn.Sequential(nn.Linear(shared_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.cell_type = nn.Sequential(nn.Linear(shared_dim, 256), nn.GELU(), nn.Linear(256, n_cell_types))
        self.species_head = nn.Sequential(nn.Linear(shared_dim, 256), nn.GELU(), nn.Linear(256, n_species))

        # Count-aware generative head. Softmax gives gene proportions; library
        # size restores count scale. theta is gene-specific dispersion.
        self.decoder = nn.Sequential(nn.Linear(shared_dim + private_dim + 256, 2048), nn.LayerNorm(2048), nn.GELU(), nn.Dropout(dropout), nn.Linear(2048, 2048), nn.GELU())
        self.mu_head = nn.Linear(2048, n_genes)
        self.masked_gene_head = nn.Linear(shared_dim + private_dim, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    @staticmethod
    def count_input(x: torch.Tensor) -> torch.Tensor:
        """Stable input representation for nonnegative count-like data."""
        if torch.any(x < 0):
            raise ValueError("CardiLearnLarge expects nonnegative count-like expression")
        return torch.log1p(x)

    def context(self, species: torch.Tensor, assay: torch.Tensor) -> torch.Tensor:
        return self.context_projection(torch.cat([self.species_embedding(species), self.assay_embedding(assay)], dim=-1))

    def encode(self, x: torch.Tensor, species: torch.Tensor, assay: torch.Tensor):
        if x.ndim != 2 or x.shape[1] != self.n_genes:
            raise ValueError(f"Expected x with shape [batch, {self.n_genes}], got {tuple(x.shape)}")
        x_input = self.count_input(x)
        gene_tokens = self.expression_projection(x_input.unsqueeze(-1)) + self.gene_embedding.unsqueeze(0)
        queries = self.program_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        programs, attention = self.gene_to_program(queries, gene_tokens, gene_tokens, need_weights=True, average_attn_weights=False)
        programs = self.cross_norm(programs + queries)
        programs = self.cross_norm2(programs + self.cross_ffn(programs))
        programs = self.program_transformer(programs)
        programs = self.program_norm(programs)
        weights = torch.softmax(self.program_pool(programs).squeeze(-1), dim=-1).unsqueeze(-1)
        molecular = torch.sum(weights * programs, dim=1)
        z_shared = self.shared_encoder(molecular)
        z_private = self.private_encoder(molecular)
        return z_shared, z_private, programs, attention

    def decode_counts(self, z_shared: torch.Tensor, z_private: torch.Tensor, species: torch.Tensor, assay: torch.Tensor, library_size: torch.Tensor | None = None):
        if library_size is None:
            library_size = torch.clamp(torch.sum(torch.exp(torch.clamp(z_private[..., :1], -5, 5)), dim=-1), min=1.0)
        context = self.context(species, assay)
        hidden = self.decoder(torch.cat([z_shared, z_private, context], dim=-1))
        proportions = torch.softmax(self.mu_head(hidden), dim=-1)
        library_size = library_size.reshape(-1, 1).to(proportions.dtype)
        mu = proportions * library_size
        theta = F.softplus(self.log_theta).unsqueeze(0).expand_as(mu) + 1e-4
        return mu, theta

    def forward(self, x: torch.Tensor, species: torch.Tensor, assay: torch.Tensor, library_size: torch.Tensor | None = None) -> CardiLearnOutput:
        z_shared, z_private, programs, attention = self.encode(x, species, assay)
        mu, theta = self.decode_counts(z_shared, z_private, species, assay, library_size)
        # The masked head predicts log1p expression. Training code should apply
        # a random gene mask to the input and evaluate this head only on masked
        # genes, avoiding trivial identity reconstruction.
        masked_prediction = self.masked_gene_head(torch.cat([z_shared, z_private], dim=-1))
        return CardiLearnOutput(
            z_shared=z_shared,
            z_private=z_private,
            maturity=self.maturity(z_shared).squeeze(-1),
            injury=self.injury(z_shared).squeeze(-1),
            cell_type=self.cell_type(z_shared),
            reconstruction=mu,
            program_tokens=programs,
            program_attention=attention,
            nb_mu=mu,
            nb_theta=theta,
            masked_prediction=masked_prediction,
            species_logits=self.species_head(z_shared),
        )

    def parameter_count(self, trainable_only: bool = True) -> int:
        params = self.parameters() if trainable_only else self.parameters()
        return sum(p.numel() for p in params if (p.requires_grad or not trainable_only))
