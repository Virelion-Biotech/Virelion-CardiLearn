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
