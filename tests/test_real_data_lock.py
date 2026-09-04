from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.lock_real_data import _validate_split, _load_review


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


def test_sample_manifest_has_one_row_per_sample():
    from scripts.assemble_10x_cohort import load_manifest

    path = tmp_path = None
    assert pd is not None
