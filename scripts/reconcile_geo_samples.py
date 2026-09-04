"""Reconcile GEO sample metadata into a reviewable CardiLearn sample manifest.

This script is intentionally conservative. It parses NCBI GEO family-SOFT metadata,
extracts only explicitly supplied fields, applies only narrowly defined mappings, and
writes unresolved values rather than guessing. It never reads expression data and never
marks a sample verified by itself.

The output is suitable for human review and subsequent conversion into the canonical
real-data sample manifest used by the lock/assembly pipeline.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from cardilearn.real_data import StudySpec, canonicalize_geo_samples, parse_geo_family_soft


OUTPUT_COLUMNS = (
    "accession",
    "study_id",
    "subject_id",
    "sample_id",
    "species",
    "assay",
    "cell_type",
    "maturation",
    "injury",
    "condition",
    "condition_status",
    "subject_confidence",
    "timepoint",
    "tissue",
    "region",
    "metadata_quality",
    "matrix_dir",
    "source_observation_id",
    "source_title",
    "condition_raw",
    "raw_characteristics",
)


def _study_from_row(row: dict[str, Any]) -> StudySpec:
    return StudySpec(
        study_id=str(row["study_id"]),
        accession=str(row["accession"]),
        source="GEO",
        species=str(row["species"]),
        modality=str(row["assay"]),
        role="candidate",
        evidence_tier="C",
        tasks=("representation_learning",),
        rationale="Candidate source pending independent metadata review.",
    )


def _normalize_text(value: Any) -> str:
    return re.sub(r"\\s+", " ", str(value or "")).strip()


def _candidate_cell_type(raw: pd.DataFrame) -> pd.Series:
    """Return only explicit cell-context values; never infer a label from tissue/title."""
    values = raw["cell_context"].map(_normalize_text)
    return values


def reconcile(soft_path: str | Path, study_id: str, accession: str, species: str, assay: str) -> pd.DataFrame:
    raw = parse_geo_family_soft(soft_path)
    study = _study_from_row(
        {
            "study_id": study_id,
            "accession": accession,
            "species": species,
            "assay": assay,
        }
    )
    canonical = canonicalize_geo_samples(raw, study)

    result = pd.DataFrame(
        {
            "accession": canonical["accession"].astype(str),
            "study_id": canonical["study_id"].astype(str),
            "subject_id": canonical["subject_id"].fillna("").map(_normalize_text),
            "sample_id": canonical["sample_id"].map(_normalize_text),
            "species": canonical["species"].astype(str),
            "assay": canonical["modality"].astype(str),
            "cell_type": _candidate_cell_type(canonical),
            "maturation": "",
            "injury": "",
            "condition": canonical["condition"].map(_normalize_text),
            "condition_status": canonical["condition_status"].map(_normalize_text),
            "subject_confidence": "unverified",
            "timepoint": canonical["timepoint"].map(_normalize_text),
            "tissue": canonical["tissue"].map(_normalize_text),
            "region": canonical["region"].map(_normalize_text),
            "metadata_quality": canonical["metadata_quality"].map(_normalize_text),
            "matrix_dir": "",
            "source_observation_id": canonical["sample_id"].map(_normalize_text),
            "source_title": canonical["title"].map(_normalize_text),
            "condition_raw": canonical["condition_raw"].map(_normalize_text),
            "raw_characteristics": canonical["raw_characteristics"].astype(str),
        }
    )

    issues = []
    for _, row in result.iterrows():
        row_issues: list[str] = []
        if not row["subject_id"]:
            row_issues.append("missing_subject_id")
        if row["condition_status"] != "controlled":
            row_issues.append("condition_unresolved")
        if not row["cell_type"]:
            row_issues.append("cell_type_unresolved")
        if not row["timepoint"]:
            row_issues.append("timepoint_unresolved")
        if not row["tissue"]:
            row_issues.append("tissue_unresolved")
        if not row["region"]:
            row_issues.append("region_unresolved")
        if row["subject_id"]:
            result.loc[row.name, "subject_confidence"] = "source_explicit_unreviewed"
        issues.append(";".join(row_issues) if row_issues else "requires_manual_review")
    result["review_issues"] = issues
    return result[list(OUTPUT_COLUMNS) + ["review_issues"]]


def _write_summary(frame: pd.DataFrame, path: Path) -> None:
    summary = {
        "samples": int(len(frame)),
        "subjects_nonempty": int(frame["subject_id"].ne("").sum()),
        "controlled_conditions": int(frame["condition_status"].eq("controlled").sum()),
        "explicit_cell_context": int(frame["cell_type"].ne("").sum()),
        "review_required": int(len(frame)),
        "verified_subjects": 0,
        "ready_for_lock": False,
        "policy": "No sample is verified or lock-eligible from this script alone.",
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile one GEO family-SOFT file into a reviewable sample manifest")
    parser.add_argument("--soft", required=True, help="GEO family-SOFT(.gz) file")
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--assay", required=True)
    parser.add_argument("--output", required=True, help="CSV output outside Git")
    parser.add_argument("--summary", required=True, help="JSON audit summary outside Git")
    args = parser.parse_args()

    frame = reconcile(args.soft, args.study_id, args.accession, args.species, args.assay)
    output = Path(args.output)
    summary = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    _write_summary(frame, summary)
    print(json.dumps({"output": str(output), "summary": str(summary), "samples": len(frame)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
