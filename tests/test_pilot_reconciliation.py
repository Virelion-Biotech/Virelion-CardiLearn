from __future__ import annotations

import pandas as pd
import pytest

from cardilearn.pilot_reconciliation import (
    audit_sample_frame,
    candidate_split_plan,
)


def _frame(n_studies: int = 3) -> pd.DataFrame:
    rows = []
    for study in range(n_studies):
        for sample in range(2):
            rows.append(
                {
                    "study_id": f"ST{study}",
                    "accession": f"GSE{10000 + study}",
                    "sample_id": f"ST{study}_S{sample}",
                    "subject_id": f"ST{study}_SUB{sample}",
                    "condition": "reference" if sample == 0 else "myocardial_injury",
                    "condition_status": "controlled",
                    "timepoint": "day 3",
                    "modality": "snRNA-seq",
                    "species": "Mus musculus",
                    "tissue": "heart",
                }
            )
    return pd.DataFrame(rows)


def test_three_study_pilot_cannot_be_locked():
    plan = candidate_split_plan(_frame(3))
    assert plan["locked"] is False
    assert plan["status"] == "candidate_only"


def test_six_studies_are_ready_for_task_specific_split_assignment():
    plan = candidate_split_plan(_frame(6))
    assert plan["locked"] is False
    assert plan["status"] == "ready_for_split_assignment"


def test_missing_subject_is_blocking():
    frame = _frame(3)
    frame.loc[0, "subject_id"] = None
    audits = audit_sample_frame(frame)
    assert audits[0].status == "blocking_missing_subject_ids"


def test_unresolved_condition_requires_reconciliation():
    frame = _frame(3)
    frame.loc[0, "condition_status"] = "unresolved"
    audits = audit_sample_frame(frame)
    assert audits[0].status == "needs_condition_reconciliation"


def test_duplicate_samples_are_blocking():
    frame = _frame(3)
    frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
    audits = audit_sample_frame(frame)
    assert audits[0].duplicate_sample_ids == 1
    assert audits[0].status == "blocking_duplicate_samples"


def test_missing_required_columns_fail_fast():
    frame = _frame(3).drop(columns=["tissue"])
    with pytest.raises(ValueError, match="missing columns"):
        audit_sample_frame(frame)
