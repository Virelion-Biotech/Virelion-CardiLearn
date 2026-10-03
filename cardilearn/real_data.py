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
import gzip
import json
import re

import pandas as pd


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
    sample_metadata_reconciled: bool = False
    split_manifest_frozen: bool = False

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
        return (
            self.ready_for_metadata_review
            and self.sample_metadata_reconciled
            and self.split_manifest_frozen
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_for_metadata_review": self.ready_for_metadata_review,
            "ready_for_lock": self.ready_for_lock,
            "sample_metadata_reconciled": self.sample_metadata_reconciled,
            "split_manifest_frozen": self.split_manifest_frozen,
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

        if "regeneration" in study.tasks and study.role not in {
            "regeneration",
            "development_regeneration_injury",
        }:
            audit.issues.append(
                AuditIssue(
                    "warning",
                    "REGENERATION_ROLE_REVIEW",
                    "regeneration is listed as a task but the study role does not explicitly identify regenerative evidence",
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


def _parse_characteristic(value: str) -> tuple[str, str]:
    """Split a GEO characteristic such as 'subject: animal-01' conservatively."""
    text = value.strip()
    if ":" not in text:
        return "raw", text
    key, raw_value = text.split(":", 1)
    key = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    return key or "raw", raw_value.strip()


def parse_geo_family_soft(path: str | Path) -> pd.DataFrame:
    """Parse sample-level fields from a GEO family-SOFT archive.

    Raw characteristic strings are retained under ``raw_characteristics`` so
    later curation can inspect exactly what the source provided. No biological
    condition is inferred here.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    characteristics: dict[str, list[str]] = {}

    def flush() -> None:
        nonlocal current, characteristics
        if current is None:
            return
        current["raw_characteristics"] = dict(characteristics)
        rows.append(current)
        current = None
        characteristics = {}

    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                flush()
                accession = line.split("=", 1)[1].strip()
                current = {"sample_id": accession}
                continue
            if current is None or not line.startswith("!Sample_"):
                continue
            field, value = line[1:].split(" = ", 1)
            key = field[len("Sample_") :].lower()
            if key.startswith("characteristics_ch1"):
                char_key, char_value = _parse_characteristic(value)
                characteristics.setdefault(char_key, []).append(char_value)
            elif key == "geo_accession":
                current["sample_id"] = value.strip()
            elif key in {"title", "source_name_ch1", "organism_ch1"}:
                current[key] = value.strip()

    flush()
    return pd.DataFrame(rows)


def _first_characteristic(
    characteristics: dict[str, list[str]],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        values = characteristics.get(key)
        if values:
            value = values[0].strip()
            if value:
                return value
    return None


def _controlled_condition(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    if normalized in {"sham", "control", "ctl", "reference", "healthy"}:
        return "reference"
    if normalized in {"mi", "myocardial infarction", "infarct", "myocardial injury"}:
        return "myocardial_injury"
    return None


def canonicalize_geo_samples(
    raw: pd.DataFrame,
    study: StudySpec,
) -> pd.DataFrame:
    """Convert parsed GEO sample fields to CardiLearn's canonical sample table."""
    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        characteristics = row.get("raw_characteristics") or {}
        subject = _first_characteristic(
            characteristics,
            (
                "subject_id",
                "subject",
                "donor_id",
                "donor",
                "animal_id",
                "animal",
                "individual",
                "patient_id",
                "patient",
            ),
        )
        condition_raw = _first_characteristic(
            characteristics,
            ("condition", "group", "phenotype", "treatment", "surgery"),
        )
        condition = _controlled_condition(condition_raw)
        timepoint = _first_characteristic(
            characteristics,
            (
                "timepoint",
                "collection_timepoint",
                "post_surgical_day",
                "day_post_surgery",
                "days_post_injury",
            ),
        )
        tissue = _first_characteristic(characteristics, ("tissue", "tissue_type"))
        region = _first_characteristic(characteristics, ("region", "tissue_region", "zone"))
        cell_context = _first_characteristic(
            characteristics,
            ("cell_type", "cell_context", "cell", "celltype"),
        )
        technical = _first_characteristic(
            characteristics,
            ("technical_replicate", "is_technical_replicate"),
        )
        technical_value = (
            None
            if technical is None
            else technical.lower() in {"1", "true", "yes", "y", "t"}
        )

        rows.append(
            {
                "study_id": study.study_id,
                "accession": study.accession,
                "sample_id": str(row.get("sample_id") or "").strip(),
                "subject_id": subject,
                "condition": condition or "",
                "condition_raw": condition_raw or "",
                "condition_status": "controlled" if condition else "unresolved",
                "timepoint": timepoint or "",
                "modality": study.modality,
                "species": study.species,
                "tissue": tissue or "",
                "region": region or "",
                "cell_context": cell_context or "",
                "is_technical_replicate": technical_value,
                "metadata_quality": "candidate_requires_review",
                "title": row.get("title", ""),
                "source_name": row.get("source_name_ch1", ""),
                "source_organism": row.get("organism_ch1", ""),
                "raw_characteristics": json.dumps(characteristics, sort_keys=True),
            }
        )

    return pd.DataFrame(rows)
