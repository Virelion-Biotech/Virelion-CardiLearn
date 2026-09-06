import pytest
import torch

from cardilearn.prototype.model import CardiLearnLarge


def test_large_model_rejects_toy_scale():
    with pytest.raises(ValueError):
        CardiLearnLarge(n_genes=1000, n_species=2, n_assays=2, n_cell_types=4,
                        gene_dim=64, n_programs=4, n_layers=1, n_heads=4)


def test_large_model_parameter_count_is_real_scale():
    model = CardiLearnLarge(
        n_genes=5000,
        n_species=3,
        n_assays=3,
        n_cell_types=8,
        gene_dim=256,
        n_programs=16,
        n_layers=2,
        n_heads=8,
        shared_dim=128,
        private_dim=64,
    )
    assert model.parameter_count() > 10_000_000
    assert model.parameter_count() == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_large_model_forward_shapes():
    model = CardiLearnLarge(
        n_genes=5000,
        n_species=2,
        n_assays=2,
        n_cell_types=5,
        gene_dim=128,
        n_programs=8,
        n_layers=1,
        n_heads=8,
        shared_dim=64,
        private_dim=32,
    )
    x = torch.poisson(torch.full((2, 5000), 0.1))
    out = model(x, torch.tensor([0, 1]), torch.tensor([0, 1]))
    assert out.z_shared.shape == (2, 64)
    assert out.reconstruction.shape == (2, 5000)
    assert out.nb_mu.shape == (2, 5000)
    assert out.nb_theta.shape == (2, 5000)
    assert out.masked_prediction.shape == (2, 5000)
    assert out.species_logits.shape == (2, 2)
