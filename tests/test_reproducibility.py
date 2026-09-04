import json

import pandas as pd
import pytest

from cardilearn.config_loader import ConfigError, validate_reproducibility_config
from cardilearn.reproducibility import (
    ReproducibilityManifest,
    dataframe_fingerprint,
    fingerprint_mapping,
    fingerprint_ids,
    load_manifest,
    make_manifest,
    save_manifest,
)


def test_mapping_fingerprint_is_order_independent():
    assert fingerprint_mapping({"b": 2, "a": 1}) == fingerprint_mapping({"a": 1, "b": 2})


def test_ids_fingerprint_preserves_order():
    assert fingerprint_ids(["a", "b"]) != fingerprint_ids(["b", "a"])


def test_dataframe_fingerprint_changes_with_values():
    left = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    right = pd.DataFrame({"x": [1, 2], "y": [3, 5]})
    assert dataframe_fingerprint(left) != dataframe_fingerprint(right)


def test_manifest_round_trip(tmp_path):
    config = {"model": "cardilearn", "seed": 42}
    manifest = make_manifest(config=config, config_name="test", seeds=[13, 42], primary_seed=42)
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    restored = load_manifest(path)
    assert restored.to_dict() == manifest.to_dict()
    assert len(restored.fingerprint()) == 64


def test_manifest_rejects_primary_seed_not_in_seed_set():
    with pytest.raises(ValueError, match="primary_seed"):
        ReproducibilityManifest(
            schema_version="1.0", project_version="0.3.0", config_name="x",
            config_fingerprint="a" * 64, data_fingerprint="b" * 64,
            split_fingerprint="c" * 64, seeds=(13, 42), primary_seed=7,
        )


def test_config_validation_requires_train_only_preprocessing():
    valid = {
        "schema_version": "1.0", "experiment": {"name": "x", "task": "classification"},
        "data": {"source": "x", "fingerprint": "a" * 64},
        "splits": {"manifest": "x", "group_column": "subject_id"},
        "reproducibility": {"seeds": [42], "primary_seed": 42, "fit_preprocessing_on_train_only": True},
    }
    assert validate_reproducibility_config(valid) == valid
    invalid = json.loads(json.dumps(valid))
    invalid["reproducibility"]["fit_preprocessing_on_train_only"] = False
    with pytest.raises(ConfigError, match="fit_preprocessing_on_train_only"):
        validate_reproducibility_config(invalid)
