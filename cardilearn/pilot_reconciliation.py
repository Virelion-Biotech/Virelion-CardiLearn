"""Real-data pilot reconciliation and lock-gate logic.

This module converts canonical candidate sample metadata into an explicit audit.
It never silently invents subject IDs, conditions, timepoints, or replicate groups.
A three-study pilot is intentionally insufficient for a final study-held-out
benchmark, so the planner emits a candidate plan rather than a fake locked split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = (
    "study_id",
    "sample_id",
    "subject_id",
    "condition",
    "timepoint",
    "modality",
    "species",
    "tissue",
)


@dataclass(frozen=True)
class StudyAudit:
    study_id: str
    samples: int
    subjects: int
    unresolved_conditions: int
    missing_subject_ids: int
    duplicate_sample_ids: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def audit_sample_frame(frame: pd.DataFrame) -> list[StudyAudit]:
    """Audit canonical pilot samples without making biological inferences."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"canonical sample table is missing columns: {missing}")

    audits: list[StudyAudit] = []
    for study_id, group in frame.groupby("study_id", sort=True):
        duplicate_sample_ids = int(group["sample_id"].duplicated().sum())
        missing_subject_ids = int(group["subject_id"].isna().sum() + group["subject_id"].eq("").sum())
        unresolved_conditions = int(
            (~group["condition_status"].eq("controlled")).sum()
            if "condition_status" in group.columns
            else group["condition"].eq("").sum()
        )

        status = "ready_for_reconciliation"
        if duplicate_sample_ids:
            status = "blocking_duplicate_samples"
        elif missing_subject_ids:
            status = "blocking_missing_subject_ids"
        elif unresolved_conditions:
            status = "needs_condition_reconciliation"

        audits.append(
            StudyAudit(
                study_id=str(study_id),
                samples=int(len(group)),
                subjects=int(group["subject_id"].replace("", pd.NA).dropna().nunique()),
                unresolved_conditions=unresolved_conditions,
                missing_subject_ids=missing_subject_ids,
                duplicate_sample_ids=duplicate_sample_ids,
                status=status,
            )
        )
    return audits


def candidate_split_plan(
    frame: pd.DataFrame,
    *,
    minimum_studies: int = 6,
) -> dict[str, Any]:
    """Return a split candidate plan; never claims a lock from too little data."""
    studies = sorted(frame["study_id"].dropna().unique().tolist())
    eligible = len(studies) >= minimum_studies

    plan: dict[str, Any] = {
        "status": "candidate_only",
        "study_count": len(studies),
        "minimum_studies_for_lock": minimum_studies,
        "studies": studies,
        "locked": False,
        "reason": "at least six independent studies are required before creating a useful multi-study train/validation/test split",
    }

    if eligible:
        # The actual split assignment remains a separate deterministic step so
        # it can incorporate task balance and CardiBench policy.
        plan["status"] = "ready_for_split_assignment"
        plan["reason"] = "sufficient study count; task-specific stratification and subject integrity must be applied before lock"

    return plan
