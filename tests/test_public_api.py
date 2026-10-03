from cardilearn import __version__
from cardilearn.models import available_models


def test_package_version_matches_release_metadata():
    assert __version__ == "0.4.1"


def test_classification_registry_is_stable():
    assert available_models("classification") == (
        "logistic_regression",
        "hist_gradient_boosting",
        "mlp",
    )


def test_regression_registry_is_stable():
    assert available_models("regression") == (
        "ridge",
        "hist_gradient_boosting",
        "mlp",
    )


def test_representation_registry_exposes_external_comparison_set():
    assert available_models("representation") == ("pca", "autoencoder", "scvi", "geneformer", "scgpt", "uce", "nicheformer", "scimilarity", "cardilearn_research")


def test_public_api_exports_are_unique():
    import cardilearn
    assert len(cardilearn.__all__) == len(set(cardilearn.__all__))


def test_external_adapter_validates_shape():
    import numpy as np
    from cardilearn.backbones import CallableEncoderAdapter
    adapter = CallableEncoderAdapter(
        name="fixture",
        encoder=lambda _: np.ones((3, 4)),
        provenance={"source": "test"},
    )
    assert adapter.encode(None).shape == (3, 4)
