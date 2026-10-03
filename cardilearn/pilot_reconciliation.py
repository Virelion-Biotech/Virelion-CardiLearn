"""Real-data pilot reconciliation and lock-gate logic.

This module converts canonical candidate sample metadata into an explicit audit.
It never silently invents subject IDs, conditions, timepoints, or replicate groups.
Study-family identities are also enforced so linked GEO series cannot masquerade
as independent held-out studies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .study_families import annotate_study_families


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
    minimum_families: int | None = None,
) -> dict[str, Any]:
    """Return a split candidate plan using independent study families.

    ``study_id`` remains the reporting unit, but ``study_family_id`` is the
    independence unit for held-out-study claims. Linked accessions are kept in
    the same future partition. Unmapped accessions are conservatively treated
    as their own family until related-series review identifies a linkage.
    """
    family_frame = annotate_study_families(frame)
    studies = sorted(family_frame["study_id"].dropna().unique().tolist())
    families = sorted(family_frame["study_family_id"].dropna().unique().tolist())
    required_families = minimum_studies if minimum_families is None else minimum_families
    eligible = len(families) >= required_families

    family_members = {
        family: sorted(
            family_frame.loc[
                family_frame["study_family_id"].eq(family), "study_id"
            ].dropna().unique().tolist()
        )
        for family in families
    }

    plan: dict[str, Any] = {
        "status": "candidate_only",
        "study_count": len(studies),
        "independent_family_count": len(families),
        "minimum_studies_for_lock": minimum_studies,
        "minimum_independent_families_for_lock": required_families,
        "studies": studies,
        "study_families": family_members,
        "locked": False,
        "reason": "at least six independent study families are required before creating a useful multi-study train/validation/test split",
    }

    if eligible:
        plan["status"] = "ready_for_split_assignment"
        plan["reason"] = "sufficient independent study families; task-specific stratification and subject integrity must be applied before lock"

    return plan
