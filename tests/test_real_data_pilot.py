from __future__ import annotations

import json

import pytest

from cardilearn.real_data import (
    audit_manifest,
    audit_sample_metadata,
    load_manifest,
)


def test_real_pilot_manifest_loads_and_has_three_candidates():
    studies = load_manifest("configs/real_data_pilot_v0_1.json")
    assert len(studies) == 3
    assert {study.accession for study in studies} == {
        "GSE185289",
        "GSE130699",
        "GSE217494",
    }


def test_real_pilot_manifest_has_no_blocking_issues():
    studies = load_manifest("configs/real_data_pilot_v0_1.json")
    audit = audit_manifest(studies)
    assert audit.blocking == []
    assert audit.ready_for_metadata_review
    assert not audit.ready_for_lock


def test_missing_subject_metadata_blocks_lock():
    issues = audit_sample_metadata(
        ["study_id", "sample_id", "condition", "timepoint", "modality", "species"]
    )
    codes = {issue.code for issue in issues}
    assert "MISSING_SUBJECT_ID" in codes


def test_duplicate_study_ids_are_rejected(tmp_path):
    manifest = {
        "studies": [
            {
                "study_id": "a",
                "accession": "GSE1",
                "source": "GEO",
                "species": "mouse",
                "modality": "snRNA-seq",
                "role": "test",
                "evidence_tier": "A",
                "tasks": ["representation"],
                "rationale": "test",
            },
            {
                "study_id": "a",
                "accession": "GSE2",
                "source": "GEO",
                "species": "pig",
                "modality": "snRNA-seq",
                "role": "test",
                "evidence_tier": "A",
                "tasks": ["representation"],
                "rationale": "test",
            },
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate study_id"):
        load_manifest(path)
