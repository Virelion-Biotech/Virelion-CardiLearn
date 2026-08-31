from cardilearn import __version__
from cardilearn.models import available_models


def test_package_version_matches_release_metadata():
    assert __version__ == "0.3.0"


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
