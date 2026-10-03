from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from cardilearn.real_data import (
    StudySpec,
    audit_manifest,
    audit_sample_metadata,
    canonicalize_geo_samples,
    load_manifest,
    parse_geo_family_soft,
)
from cardilearn.study_families import annotate_study_families, accession_to_family


def test_real_pilot_manifest_loads_and_has_expanded_candidate_set():
    studies = load_manifest("configs/real_data_pilot_v0_1.json")
    assert len(studies) == 8
    assert {study.accession for study in studies} == {
        "GSE185289", "GSE130699", "GSE217494", "GSE153480",
        "GSE135310", "GSE106472", "GSE216211", "GSE269054",
    }


def test_real_pilot_manifest_has_no_blocking_issues_but_is_not_locked():
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
                "study_id": "a", "accession": "GSE1", "source": "GEO",
                "species": "mouse", "modality": "snRNA-seq", "role": "test",
                "evidence_tier": "A", "tasks": ["representation"], "rationale": "test",
            },
            {
                "study_id": "a", "accession": "GSE2", "source": "GEO",
                "species": "pig", "modality": "snRNA-seq", "role": "test",
                "evidence_tier": "A", "tasks": ["representation"], "rationale": "test",
            },
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate study_id"):
        load_manifest(path)


def test_geo_family_soft_parser_preserves_raw_characteristics(tmp_path):
    content = (
        "^SAMPLE = GSM000001\n"
        "!Sample_geo_accession = GSM000001\n"
        "!Sample_title = Example MI sample\n"
        "!Sample_characteristics_ch1 = subject: mouse-01\n"
        "!Sample_characteristics_ch1 = condition: myocardial infarction\n"
        "!Sample_characteristics_ch1 = timepoint: day 3\n"
        "!Sample_characteristics_ch1 = tissue: left ventricle\n"
    )
    path = tmp_path / "family.soft.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(content)

    frame = parse_geo_family_soft(path)
    assert len(frame) == 1
    assert frame.loc[0, "sample_id"] == "GSM000001"
    assert frame.loc[0, "raw_characteristics"]["subject"] == ["mouse-01"]


def test_canonicalization_only_controls_known_condition_terms():
    study = StudySpec(
        study_id="study:test", accession="GSE999999", source="GEO",
        species="Mus musculus", modality="snRNA-seq",
        role="development_regeneration_injury", evidence_tier="A",
        tasks=("representation", "injury"), rationale="test",
    )
    raw = pd.DataFrame([
        {"sample_id": "GSM1", "raw_characteristics": {
            "subject": ["M1"], "condition": ["MI"], "timepoint": ["day 3"], "tissue": ["heart"]}},
        {"sample_id": "GSM2", "raw_characteristics": {
            "subject": ["M2"], "condition": ["ARP1MIP28-P35"], "timepoint": ["P35"], "tissue": ["heart"]}},
    ])

    canonical = canonicalize_geo_samples(raw, study)
    assert canonical.loc[0, "condition"] == "myocardial_injury"
    assert canonical.loc[0, "condition_status"] == "controlled"
    assert canonical.loc[1, "condition"] == ""
    assert canonical.loc[1, "condition_status"] == "unresolved"


def test_linked_geo_series_share_one_study_family():
    mapping = accession_to_family()
    assert mapping["GSE130699"] == mapping["GSE153480"]

    frame = pd.DataFrame([
        {"study_id": "mouse_a", "accession": "GSE130699"},
        {"study_id": "mouse_b", "accession": "GSE153480"},
        {"study_id": "pig_a", "accession": "GSE185289"},
    ])
    annotated = annotate_study_families(frame)
    assert annotated.loc[0, "study_family_id"] == annotated.loc[1, "study_family_id"]
    assert annotated.loc[2, "study_family_id"] != annotated.loc[0, "study_family_id"]
