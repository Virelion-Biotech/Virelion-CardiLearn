"""Unit tests for torch training objectives and the species adversary."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from cardilearn import multimodal_losses as mm
from cardilearn.prototype import losses
from cardilearn.prototype.species_adversary import GradientReversal, SpeciesAdversary


def test_cross_modal_infonce_rewards_matched_pairs():
    torch.manual_seed(0)
    z = torch.randn(8, 4)
    matched = mm.cross_modal_infonce(z, z)
    shuffled = mm.cross_modal_infonce(z, z[torch.randperm(8)])
    assert matched < shuffled
    for a, b, kw in [(z, z[:, :2], {}), (z[:1], z[:1], {}), (z, z, {"temperature": 0.0})]:
        with pytest.raises(ValueError):
            mm.cross_modal_infonce(a, b, **kw)


def test_orthogonality_consistency_and_uncertainty():
    torch.manual_seed(1)
    z = torch.randn(16, 3)
    assert mm.shared_private_orthogonality(z, z) > mm.shared_private_orthogonality(z, torch.randn(16, 3))
    with pytest.raises(ValueError):
        mm.shared_private_orthogonality(z, z[:4])
    assert mm.modality_consistency([z]).item() == 0.0
    assert mm.modality_consistency([z, z, z]).item() == pytest.approx(0.0, abs=1e-6)
    assert mm.modality_consistency([z, -z]).item() == pytest.approx(2.0, abs=1e-5)
    sigma = torch.ones(4)
    assert mm.uncertainty_regularization(sigma, torch.full((4,), 2.0)).item() == pytest.approx(2.0)
    with pytest.raises(ValueError):
        mm.uncertainty_regularization(sigma, torch.ones(3))


def test_negative_binomial_loss_reductions_and_guards():
    mu = torch.tensor([1.0, 5.0])
    theta = torch.tensor([2.0, 2.0])
    counts = torch.tensor([1.0, 4.0])
    per = losses.negative_binomial_loss(mu, theta, counts, reduction="none")
    assert per.shape == (2,) and torch.all(per > 0)
    assert losses.negative_binomial_loss(mu, theta, counts).item() == pytest.approx(per.mean().item())
    assert losses.negative_binomial_loss(mu, theta, counts, reduction="sum").item() == pytest.approx(per.sum().item())
    with pytest.raises(ValueError):
        losses.negative_binomial_loss(mu, theta, -counts)
    with pytest.raises(ValueError):
        losses.negative_binomial_loss(-mu, theta, counts)


def test_masked_and_auxiliary_losses():
    target = torch.tensor([[0.0, 3.0]])
    pred = torch.log1p(target)
    assert losses.masked_gene_loss(pred, target, torch.ones_like(target)).item() == pytest.approx(0.0)
    assert losses.masked_gene_loss(pred + 1, target, torch.zeros_like(target)).item() == 0.0
    assert losses.reconstruction_loss(pred, pred).item() == 0.0
    z = torch.randn(10, 4)
    assert losses.invariance_loss(z, 2 * z).item() == pytest.approx(0.0, abs=1e-6)
    collapsed = torch.zeros(10, 4)
    assert losses.variance_loss(collapsed).item() > 0.9
    assert losses.covariance_loss(collapsed).item() == 0.0
    assert losses.anti_collapse_loss(collapsed).item() == pytest.approx(0.5 * losses.variance_loss(collapsed).item())
    a, b = torch.tensor([2.0]), torch.tensor([0.0])
    assert losses.maturation_pair_loss(a, b) < losses.maturation_pair_loss(b, a)
    assert losses.regeneration_rank_loss(a, b, weight=torch.tensor([0.0])).item() == 0.0
    assert losses.regeneration_rank_loss(a, b) == losses.maturation_pair_loss(a, b)
    assert losses.maturation_absolute_loss(a, a).item() == 0.0
    assert losses.injury_loss(torch.tensor([10.0]), torch.tensor([1])).item() < 0.01
    logits = torch.tensor([[10.0, -10.0]])
    assert losses.cell_type_loss(logits, torch.tensor([0])).item() < 0.01
    assert losses.species_adversarial_loss(logits, torch.tensor([1])).item() > 10


def test_gradient_reversal_and_species_adversary():
    x = torch.ones(3, requires_grad=True)
    out = GradientReversal(0.5)(x)
    assert torch.equal(out, x)
    out.sum().backward()
    assert x.grad.tolist() == [-0.5, -0.5, -0.5]
    with pytest.raises(ValueError):
        GradientReversal(-1.0)
    adversary = SpeciesAdversary(latent_dim=4, n_species=3, hidden_dim=8)
    assert adversary(torch.randn(5, 4)).shape == (5, 3)
    with pytest.raises(ValueError):
        adversary(torch.randn(4))
    with pytest.raises(ValueError):
        SpeciesAdversary(latent_dim=4, n_species=1)


def _toy_loader(n_genes: int = 12, n: int = 8):
    from torch.utils.data import DataLoader

    g = torch.Generator().manual_seed(0)
    rows = [
        {
            "x": torch.poisson(torch.full((n_genes,), 3.0), generator=g),
            "species": i % 2,
            "assay": 0,
            "cell_type": i % 3,
            "maturation": float(i) / n,
            "injury": float(i % 2),
        }
        for i in range(n)
    ]
    return DataLoader(rows, batch_size=4)


@pytest.mark.parametrize("large", [False, True])
def test_train_one_epoch_and_encode(large):
    from cardilearn.prototype.model import CardiLearnLarge, CardiLearnProto
    from cardilearn.prototype.train import TrainConfig, encode_dataset, seed_torch, train_one_epoch

    seed_torch(0)
    if large:
        n_genes = 5000  # the large model refuses toy-scale inputs
        model = CardiLearnLarge(n_genes, 2, 1, 3, gene_dim=16, n_programs=4, n_layers=1, n_heads=2,
                                ff_mult=2, shared_dim=8, private_dim=4, dropout=0.0)
    else:
        n_genes = 12
        model = CardiLearnProto(n_genes, 2, 1, 3, gene_dim=16, n_programs=4, shared_dim=8, private_dim=4)
    loader = _toy_loader(n_genes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_one_epoch(model, loader, optimizer, torch.device("cpu"), TrainConfig(mask_fraction=0.2))
    assert set(metrics) >= {"loss", "recon", "masked", "species_adversarial"}
    assert all(math.isfinite(v) for v in metrics.values())
    if not large:
        assert metrics["masked"] == 0.0 and metrics["species_adversarial"] == 0.0
    encoded = encode_dataset(model, loader, torch.device("cpu"))
    assert encoded["z_shared"].shape == (8, 8)
    assert encoded["cell_type"].shape == (8, 3)


def test_mask_genes_rejects_bad_fraction():
    from cardilearn.prototype.train import _mask_genes

    masked, mask = _mask_genes(torch.ones(2, 50), 0.5)
    assert torch.equal(masked[mask], torch.zeros(int(mask.sum())))
    with pytest.raises(ValueError):
        _mask_genes(torch.ones(2, 2), 1.0)
