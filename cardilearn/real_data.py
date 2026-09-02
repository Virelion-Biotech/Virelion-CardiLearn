"""Metadata-first real-data pilot contracts for CardiLearn.

This module deliberately stops before expression-matrix training. It turns a
curated pilot manifest plus recovered sample metadata into validation findings
and task eligibility. A dataset cannot become a locked benchmark merely because
its accession is known.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json


@dataclass(frozen=True)
class StudySpec:
    study_id: str
    accession: str
    source: str
    species: str
    modality: str
    role: str
    evidence_tier: str
    tasks: tuple[str, ...]
    rationale: str
    source_urls: tuple[str, ...] = ()
    metadata_required: tuple[str, ...] = (
        "study_id",
        "subject_id",
        "sample_id",
        "condition",
        "timepoint",
        "modality",
        "species",
    )


@dataclass
class AuditIssue:
    severity: str
    code: str
    message: str
    study_id: str | None = None


@dataclass
class PilotAudit:
    studies: list[StudySpec]
    issues: list[AuditIssue] = field(default_factory=list)

    @property
    def blocking(self) -> list[AuditIssue]:
        return [issue for issue in self.issues if issue.severity == "blocking"]

    @property
    def warnings(self) -> list[AuditIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ready_for_metadata_review(self) -> bool:
        return not self.blocking

    @property
    def ready_for_lock(self) -> bool:
        return self.ready_for_metadata_review and not any(
            issue.code.startswith("MISSING_") for issue in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_for_metadata_review": self.ready_for_metadata_review,
            "ready_for_lock": self.ready_for_lock,
            "studies": [
                {
                    "study_id": study.study_id,
                    "accession": study.accession,
                    "source": study.source,
                    "species": study.species,
                    "modality": study.modality,
                    "role": study.role,
                    "evidence_tier": study.evidence_tier,
                    "tasks": list(study.tasks),
                    "rationale": study.rationale,
                    "source_urls": list(study.source_urls),
                    "metadata_required": list(study.metadata_required),
                }
                for study in self.studies
            ],
            "issues": [issue.__dict__ for issue in self.issues],
        }


def load_manifest(path: str | Path) -> list[StudySpec]:
    """Load a JSON pilot manifest into typed study specifications."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("studies")
    if not isinstance(rows, list) or not rows:
        raise ValueError("pilot manifest must contain a non-empty 'studies' list")

    studies: list[StudySpec] = []
    seen_ids: set[str] = set()
    for row in rows:
        required = {
            "study_id",
            "accession",
            "source",
            "species",
            "modality",
            "role",
            "evidence_tier",
            "tasks",
            "rationale",
        }
        missing = required.difference(row)
        if missing:
            raise ValueError(f"study row missing keys: {sorted(missing)}")
        if row["study_id"] in seen_ids:
            raise ValueError(f"duplicate study_id: {row['study_id']}")
        seen_ids.add(row["study_id"])
        studies.append(
            StudySpec(
                study_id=str(row["study_id"]),
                accession=str(row["accession"]),
                source=str(row["source"]),
                species=str(row["species"]),
                modality=str(row["modality"]),
                role=str(row["role"]),
                evidence_tier=str(row["evidence_tier"]),
                tasks=tuple(str(x) for x in row["tasks"]),
                rationale=str(row["rationale"]),
                source_urls=tuple(str(x) for x in row.get("source_urls", [])),
            )
        )
    return studies


def audit_manifest(studies: Iterable[StudySpec]) -> PilotAudit:
    """Validate the pilot's scientific role assignments before matrix materialization."""
    studies = list(studies)
    audit = PilotAudit(studies=studies)

    accessions: set[str] = set()
    for study in studies:
        acc = study.accession.upper()
        if acc in accessions:
            audit.issues.append(
                AuditIssue(
                    "blocking",
                    "DUPLICATE_ACCESSION",
                    f"accession {acc} appears more than once",
                    study.study_id,
                )
            )
        accessions.add(acc)

        if not study.tasks:
            audit.issues.append(
                AuditIssue(
                    "blocking",
                    "NO_TASKS",
                    "study has no declared learning tasks",
                    study.study_id,
                )
            )

        if study.role == "regeneration" and study.evidence_tier not in {"A", "B"}:
            audit.issues.append(
                AuditIssue(
                    "blocking",
                    "WEAK_REGENERATION_ROLE",
                    "regeneration-role studies require tier A or B evidence",
                    study.study_id,
                )
            )

        if "regeneration" in study.tasks and study.role != "regeneration":
            audit.issues.append(
                AuditIssue(
                    "warning",
                    "REGENERATION_ROLE_REVIEW",
                    "regeneration is listed as a task but the study role is not 'regeneration'; manual review required",
                    study.study_id,
                )
            )

        if study.source == "GEO" and not acc.startswith("GSE"):
            audit.issues.append(
                AuditIssue(
                    "blocking",
                    "INVALID_GEO_ACCESSION",
                    "GEO source must use a GSE accession",
                    study.study_id,
                )
            )

    return audit


def audit_sample_metadata(
    metadata_columns: Iterable[str],
    *,
    required_columns: Iterable[str] = (
        "study_id",
        "subject_id",
        "sample_id",
        "condition",
        "timepoint",
        "modality",
        "species",
    ),
) -> list[AuditIssue]:
    """Check recovered metadata columns without guessing missing biological labels."""
    columns = set(metadata_columns)
    issues: list[AuditIssue] = []
    for name in required_columns:
        if name not in columns:
            issues.append(
                AuditIssue(
                    "blocking",
                    f"MISSING_{name.upper()}",
                    f"required metadata column '{name}' has not been recovered",
                )
            )
    return issues
