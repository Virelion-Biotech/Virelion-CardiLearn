from __future__ import annotations

import numpy as np

from cardilearn.benchmark_suite import baseline_names, benchmark_manifest, build_supervised_baseline, fit_autoencoder


def test_baseline_registry_is_stable():
    assert baseline_names() == ("pca_linear", "mlp", "autoencoder", "cardilearn")
    manifest = benchmark_manifest()
    assert [item["name"] for item in manifest] == list(baseline_names())


def test_supervised_baselines_fit_on_numeric_data():
    x = np.arange(40, dtype=float).reshape(10, 4)
    y = np.array([0, 1] * 5)
    model = build_supervised_baseline("pca_linear", task="classification", n_components=2)
    model.fit(x, y)
    assert model.predict(x).shape == (10,)


def test_autoencoder_returns_latent_representation():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(12, 6))
    _, encode = fit_autoencoder(x, latent_dim=2, hidden_layer_sizes=(5,), random_state=42)
    z = encode(x[:3])
    assert z.shape == (3, 2)


def test_frozen_embedding_benchmark_uses_grouped_classification():
    import numpy as np
    from cardilearn.representation_benchmark import benchmark_embedding
    z = np.array([[0.0,0.0],[0.1,0.0],[1.0,1.0],[1.1,1.0],[0.0,0.2],[1.2,0.8]])
    y = np.array([0,0,1,1,0,1])
    groups = np.array(["a","b","c","d","e","f"])
    result = benchmark_embedding(z,y,model_name="fixture",task="classification",groups=groups,n_splits=3)
    assert result.primary_metric == "auroc"
    assert len(result.fold_results) == 3


def test_grouped_permutation_rejects_inconsistent_group_labels():
    import numpy as np
    from cardilearn.representation_benchmark import permutation_null_auroc
    z = np.random.default_rng(0).normal(size=(6,4))
    y = np.array([0,1,1,0,1,0])
    groups = np.array(["a","a","b","c","d","e"])
    import pytest
    with pytest.raises(ValueError, match="exactly one class label"):
        permutation_null_auroc(z,y,groups=groups,n_permutations=2,n_splits=2)


def test_embedding_benchmark_can_aggregate_cell_embeddings_to_biological_groups():
    import numpy as np
    from cardilearn.representation_benchmark import aggregate_embeddings_by_group
    z = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    y = np.array([0,0,1,1])
    groups = np.array(["s1","s1","s2","s2"])
    zg, yg, gg = aggregate_embeddings_by_group(z, groups, y)
    assert zg.shape == (2, 2)
    assert yg.tolist() == [0, 1]
    assert gg.tolist() == ["s1", "s2"]


def test_bootstrap_metric_can_resample_biological_groups():
    import numpy as np
    from cardilearn.representation_benchmark import bootstrap_metric
    y = np.array([0, 0, 1, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    groups = np.array(["s1","s1","s2","s2","s3","s3"])
    result = bootstrap_metric(y, score, groups=groups, n_bootstrap=50, seed=1)
    assert result["n_valid"] > 0
    assert result["ci95_low"] <= result["ci95_high"]
