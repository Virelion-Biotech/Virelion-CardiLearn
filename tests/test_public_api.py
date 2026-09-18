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
    assert available_models("representation") == ("pca", "autoencoder", "scvi", "geneformer", "scgpt", "uce", "cardilearn_research")
