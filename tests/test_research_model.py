from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cardilearn.research_model import CardiLearnResearch, GRNProgramRouter, GeneValueEncoder, ProgramBackbone


def test_fractional_and_continuous_gene_encoders():
    x = torch.poisson(torch.full((3, 64), 0.2))
    continuous = GeneValueEncoder(64, 32, value_style="continuous")
    ranked = GeneValueEncoder(64, 32, value_style="rank")
    assert continuous(x).shape == (3, 64, 32)
    assert ranked(x).shape == (3, 64, 32)


def test_router_shape_and_program_mass():
    x = torch.randn(2, 64, 32)
    router = GRNProgramRouter(64, 8, 32, chunk_size=16)
    programs, mass = router(x)
    assert programs.shape == (2, 8, 32)
    assert mass.shape == (2, 8)
    assert torch.allclose(mass.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_transformer_program_backbone():
    backbone = ProgramBackbone(32, layers=1, heads=4)
    y = backbone(torch.randn(2, 8, 32))
    assert y.shape == (2, 8, 32)


def test_research_model_factorized_decoder():
    model = CardiLearnResearch(
        n_genes=64,
        n_species=2,
        n_assays=2,
        n_cell_types=4,
        gene_dim=32,
        n_programs=8,
        n_layers=1,
        n_heads=4,
        shared_dim=24,
        private_dim=12,
        decoder_dim=16,
        router_chunk_size=16,
        species_adversarial_strength=0.25,
    )
    x = torch.poisson(torch.full((2, 64), 0.2))
    out = model(x, torch.tensor([0, 1]), torch.tensor([0, 1]))
    assert out.z_shared.shape == (2, 24)
    assert out.z_private.shape == (2, 12)
    assert out.reconstruction_mu.shape == (2, 64)
    assert out.reconstruction_theta.shape == (2, 64)
    assert out.masked_prediction.shape == (2, 64)
    assert out.species_logits.shape == (2, 2)
    assert torch.isfinite(out.reconstruction_mu).all()
    assert torch.isfinite(out.reconstruction_theta).all()


def test_factorized_decoder_has_no_dense_hidden_to_gene_matrix():
    model = CardiLearnResearch(
        n_genes=5000,
        n_species=1,
        n_assays=1,
        n_cell_types=2,
        gene_dim=32,
        n_programs=8,
        n_layers=1,
        n_heads=4,
        shared_dim=24,
        private_dim=12,
        decoder_dim=16,
    )
    assert model.decoder.gene_projection.shape == (5000, 16)
