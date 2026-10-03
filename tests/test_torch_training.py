from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cardilearn.objectives import ObjectiveSchedule, ObjectiveStage, ObjectiveWeights, masked_log1p_loss, negative_binomial_nll, vicreg_loss
from cardilearn.torch_training import CategoryEncoder, TorchTrainConfig, random_gene_mask


def test_category_encoder_uses_zero_for_unknown():
    encoder = CategoryEncoder(["a", "b"])
    assert encoder.encode(["b", "unknown"]).tolist() == [2, 0]
    assert encoder.to_dict()["<UNK>"] == 0


def test_mask_has_signal_and_preserves_shape():
    x = torch.ones(4, 16)
    mask = random_gene_mask(x, 0.15)
    assert mask.shape == x.shape
    assert mask.any(dim=1).all()


def test_nb_and_masked_losses_are_finite():
    counts = torch.poisson(torch.full((3, 8), 0.3))
    mu = torch.full_like(counts, 0.5) + 1e-3
    theta = torch.ones_like(counts)
    assert torch.isfinite(negative_binomial_nll(mu, theta, counts))
    prediction = torch.log1p(mu)
    mask = torch.zeros_like(counts, dtype=torch.bool)
    mask[:, 0] = True
    assert torch.isfinite(masked_log1p_loss(prediction, counts, mask))


def test_vicreg_rejects_singleton_batch():
    with pytest.raises(ValueError):
        vicreg_loss(torch.randn(1, 4), torch.randn(1, 4))


def test_schedule_is_deterministic():
    schedule = ObjectiveSchedule((
        ObjectiveStage("pretrain", ObjectiveWeights(reconstruction=1.0, masked=1.0)),
        ObjectiveStage("state", ObjectiveWeights(maturation=1.0)),
    ))
    assert schedule.at_epoch(1).name == "pretrain"
    assert schedule.at_epoch(20).name == "state"
    assert TorchTrainConfig(epochs=2).epochs == 2
