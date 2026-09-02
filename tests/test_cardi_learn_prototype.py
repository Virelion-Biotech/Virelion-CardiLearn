from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from cardilearn.prototype.augmentations import make_two_views
from cardilearn.prototype.data import select_genes_train_only
from cardilearn.prototype.model import CardiLearnProto
from cardilearn.prototype.splits import assert_no_hierarchy_leakage, assign_split, study_split
from cardilearn.prototype.synthetic import SyntheticConfig, generate_synthetic_cardiac_data


def test_synthetic_generation_is_deterministic():
    cfg = SyntheticConfig(
        n_studies=4,
        subjects_per_study=2,
        samples_per_subject=2,
        cells_per_sample=5,
        n_genes=20,
    )
    x1, m1 = generate_synthetic_cardiac_data(cfg)
    x2, m2 = generate_synthetic_cardiac_data(cfg)
    assert np.array_equal(x1, x2)
    assert m1.equals(m2)


def test_hierarchical_split_has_no_leakage():
    cfg = SyntheticConfig(
        n_studies=6,
        subjects_per_study=2,
        samples_per_subject=2,
        cells_per_sample=3,
        n_genes=20,
    )
    _, metadata = generate_synthetic_cardiac_data(cfg)
    splits = study_split(metadata, seed=1)
    assigned = assign_split(metadata, splits)
    assert_no_hierarchy_leakage(assigned)
    assert assigned["_split"].notna().all()


def test_gene_selection_uses_training_only():
    x = np.zeros((10, 4), dtype=np.float32)
    x[:5, 0] = np.arange(5)
    x[:5, 1] = 1
    x[5:, 3] = np.arange(5) * 100
    import pandas as pd
    metadata = pd.DataFrame({
        "_split": ["train"] * 5 + ["test"] * 5,
        "study_id": ["A"] * 5 + ["B"] * 5,
        "subject_id": [f"s{i}" for i in range(10)],
        "sample_id": [f"m{i}" for i in range(10)],
    })
    selected = select_genes_train_only(x, metadata, 2)
    assert 3 not in selected


def test_model_forward_and_two_views():
    torch.manual_seed(42)
    model = CardiLearnProto(
        n_genes=20,
        n_species=3,
        n_assays=2,
        n_cell_types=3,
        gene_dim=16,
        n_programs=4,
        shared_dim=8,
        private_dim=4,
    )
    x = torch.randn(5, 20)
    species = torch.tensor([0, 1, 2, 0, 1])
    assay = torch.tensor([0, 1, 0, 1, 0])
    x1, x2 = make_two_views(x, 0.15)
    out = model(x1, species, assay)
    out2 = model(x2, species, assay)
    assert out.z_shared.shape == (5, 8)
    assert out.z_private.shape == (5, 4)
    assert out.reconstruction.shape == (5, 20)
    assert out.program_tokens.shape == (5, 4, 16)
    assert out2.z_shared.shape == out.z_shared.shape
