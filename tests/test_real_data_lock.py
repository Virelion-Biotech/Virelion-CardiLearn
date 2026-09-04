from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.lock_real_data import _load_review, _validate_split


def test_split_must_cover_families_exactly_once():
    families = {"f1", "f2", "f3", "f4", "f5", "f6"}
    split = {
        "unit": "study_family_id",
        "train": ["f1", "f2", "f3"],
        "validation": ["f4"],
        "test": ["f5"],
    }
    with pytest.raises(ValueError, match="every independent study family"):
        _validate_split(split, families)


def test_review_requires_explicit_approval(tmp_path):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"approved": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="approved=true"):
        _load_review(path)


def test_review_requires_reviewer_and_timestamp(tmp_path):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"approved": True, "assignments": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer"):
        _load_review(path)


def test_sample_manifest_rejects_duplicate_sample_rows(tmp_path):
    from scripts.assemble_10x_cohort import load_manifest

    row = {
        "accession": "GSE1",
        "study_id": "study1",
        "subject_id": "subject1",
        "sample_id": "sample1",
        "species": "Mus musculus",
        "assay": "scRNA-seq",
        "cell_type": "cardiomyocyte",
        "maturation": "neonatal",
        "injury": "MI",
        "condition": "injured",
        "condition_status": "controlled",
        "subject_confidence": "verified",
        "matrix_dir": "/tmp/matrix1",
    }
    frame = pd.DataFrame([row, row])
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="sample_id must occur once"):
        load_manifest(path)
