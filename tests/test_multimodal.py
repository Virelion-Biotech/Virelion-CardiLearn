import numpy as np
import pandas as pd
import pytest
import torch
from cardilearn.fusion import align_modalities, concatenate_embeddings
from cardilearn.modalities import OmicsMatrix, Waveform
from cardilearn.multimodal import CardiLearnX, SignalPatchEncoder, cosine_alignment_loss, modality_dropout


def test_align_modalities_preserves_common_subjects():
    a = pd.DataFrame({"sample_id":["s1","s2"],"age":[10,20]})
    b = pd.DataFrame({"sample_id":["s2","s3"],"rate":[2,3]})
    out = align_modalities({"clinical":a, "ecg":b})
    assert out["sample_id"].tolist() == ["s2"] and "rate" in out


def test_embeddings_require_identical_order():
    with pytest.raises(ValueError):
        concatenate_embeddings({"a":(["s1","s2"], np.ones((2,2))), "b":(["s2","s1"], np.ones((2,3)))})


def test_modality_containers_validate_shape():
    x = OmicsMatrix(np.ones((2,3)), ["s1","s2"], ["g1","g2","g3"])
    assert x.log1p().matrix.shape == (2,3)
    w = Waveform(np.ones((2,1000)), 500, "s1")
    assert w.duration_s == 2


def test_signal_encoder_shape():
    model = SignalPatchEncoder(in_channels=2, d_model=64, patch_size=16, depth=1, heads=4)
    x = torch.randn(3, 2, 128)
    assert model(x).shape == (3, 64)


def test_multimodal_shared_state_and_missing_modality():
    model = CardiLearnX({"rna": 32, "ecg": 32, "mea": 32}, fusion_dim=48, private_dim=16, n_phenotypes=4)
    embeddings = {m: torch.randn(5, 32) for m in ("rna", "ecg", "mea")}
    available = {"rna": torch.ones(5, dtype=torch.bool), "ecg": torch.tensor([1, 1, 0, 0, 1], dtype=torch.bool), "mea": torch.ones(5, dtype=torch.bool)}
    out = model(embeddings, available=available)
    assert out.z_shared.shape == (5, 48)
    assert out.private["rna"].shape == (5, 16)
    assert out.phenotype.shape == (5, 4)
    assert out.uncertainty.shape == (5,)
    assert out.availability.shape == (5, 3)


def test_alignment_requires_pairs():
    a = torch.randn(4, 16)
    b = torch.randn(4, 16)
    loss = cosine_alignment_loss(a, b)
    assert torch.isfinite(loss)
    with pytest.raises(ValueError):
        cosine_alignment_loss(a, torch.randn(3, 16))


def test_modality_dropout_preserves_at_least_one():
    x = {m: torch.randn(2, 8) for m in ("rna", "ecg", "mea")}
    out = modality_dropout(x, probability=0.999)
    assert len(out) >= 1
    assert set(out).issubset(set(x))
