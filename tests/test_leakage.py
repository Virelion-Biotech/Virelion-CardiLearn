from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cardilearn.leakage import (
    LeakageError,
    assert_no_leakage,
    audit_hierarchy,
    exact_cross_split_feature_collisions,
    freeze_split_manifest,
    verify_frozen_split_manifest,
)
from cardilearn.prototype.data import select_genes_train_only


def clean_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"study_family_id": "fam_a", "study_id": "GSE1", "subject_id": "A1", "sample_id": "A1s", "_split": "train"},
            {"study_family_id": "fam_b", "study_id": "GSE2", "subject_id": "B1", "sample_id": "B1s", "_split": "validation"},
            {"study_family_id": "fam_c", "study_id": "GSE3", "subject_id": "C1", "sample_id": "C1s", "_split": "test"},
        ]
    )


def test_clean_hierarchy_has_no_findings():
    assert audit_hierarchy(clean_frame()) == []
    assert_no_leakage(clean_frame())


def test_repeated_sample_id_within_partition_is_allowed_for_cells():
    frame = pd.concat([clean_frame(), clean_frame().iloc[[0]]], ignore_index=True)
    assert audit_hierarchy(frame) == []


def test_subject_cross_split_is_blocking():
    frame = clean_frame()
    frame.loc[len(frame)] = {
        "study_family_id": "fam_x", "study_id": "GSE4", "subject_id": "A1", "sample_id": "A1s2", "_split": "test",
    }
    findings = audit_hierarchy(frame)
    assert any(f.code == "SUBJECT_ID_CROSS_SPLIT" for f in findings)
    with pytest.raises(LeakageError):
        assert_no_leakage(frame)


def test_study_family_cross_split_is_blocking():
    frame = clean_frame()
    frame.loc[len(frame)] = {
        "study_family_id": "fam_a", "study_id": "GSE9", "subject_id": "A2", "sample_id": "A2s", "_split": "test",
    }
    findings = audit_hierarchy(frame)
    assert any(f.code == "STUDY_FAMILY_ID_CROSS_SPLIT" for f in findings)


def test_missing_subject_is_blocking():
    frame = clean_frame()
    frame.loc[1, "subject_id"] = ""
    findings = audit_hierarchy(frame)
    assert any(f.code == "MISSING_SUBJECT_ID" for f in findings)


def test_missing_partition_is_blocking():
    frame = clean_frame().query("_split != 'validation'").reset_index(drop=True)
    findings = audit_hierarchy(frame)
    assert any(f.code == "MISSING_PARTITION" for f in findings)


def test_frozen_manifest_round_trip_and_tamper_detection():
    frame = clean_frame()
    manifest = freeze_split_manifest(frame)
    assert len(manifest["records"]) == 3
    verify_frozen_split_manifest(frame, manifest)

    tampered = frame.copy()
    tampered.loc[2, "_split"] = "train"
    with pytest.raises(LeakageError, match="differ from frozen manifest"):
        verify_frozen_split_manifest(tampered, manifest)


def test_manifest_hash_tamper_detection():
    frame = clean_frame()
    manifest = freeze_split_manifest(frame)
    manifest["records"][0]["split"] = "test"
    with pytest.raises(LeakageError, match="hash"):
        verify_frozen_split_manifest(frame, manifest)


def test_exact_cross_split_feature_collisions_are_adversarial_not_within_split():
    X = np.array([[1, 2], [1, 2], [3, 4]], dtype=np.float32)
    clear = exact_cross_split_feature_collisions(X, ["train", "train", "test"])
    assert clear["status"] == "clear"

    leaking = exact_cross_split_feature_collisions(X, ["train", "test", "test"])
    assert leaking["status"] == "blocking"
    assert leaking["cross_split_collision_count"] == 1


def test_feature_selection_uses_training_rows_only():
    X = np.array([
        [0.0, 1.0, 0.0],
        [0.1, 1.1, 0.0],
        [0.2, 1.2, 0.0],
        [100.0, 0.0, 0.0],
        [-100.0, 0.0, 0.0],
    ], dtype=np.float32)
    meta = pd.DataFrame([
        {"study_id": "S1", "subject_id": "A", "sample_id": "A1", "_split": "train"},
        {"study_id": "S1", "subject_id": "B", "sample_id": "B1", "_split": "train"},
        {"study_id": "S2", "subject_id": "C", "sample_id": "C1", "_split": "train"},
        {"study_id": "S3", "subject_id": "D", "sample_id": "D1", "_split": "validation"},
        {"study_id": "S4", "subject_id": "E", "sample_id": "E1", "_split": "test"},
    ])
    selected = select_genes_train_only(X, meta, n_genes=1)
    assert selected.tolist() == [1]


def test_feature_selection_refuses_hierarchy_leak():
    X = np.eye(3, dtype=np.float32)
    meta = pd.DataFrame([
        {"study_id": "S1", "subject_id": "A", "sample_id": "A1", "_split": "train"},
        {"study_id": "S2", "subject_id": "A", "sample_id": "A2", "_split": "test"},
        {"study_id": "S3", "subject_id": "C", "sample_id": "C1", "_split": "validation"},
    ])
    with pytest.raises(LeakageError):
        select_genes_train_only(X, meta, n_genes=1)
