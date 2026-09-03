"""Study-family registry utilities for leakage-safe external evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path("configs/study_family_registry_v0_1.json")


def load_study_family_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load a versioned study-family registry."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload.get("families"), dict):
        raise ValueError("study-family registry must contain a 'families' object")
    return payload


def accession_to_family(
    path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, str]:
    """Return an accession -> biological study-family mapping."""
    payload = load_study_family_registry(path)
    mapping: dict[str, str] = {}
    for family_id, family in payload["families"].items():
        accessions = family.get("accessions", [])
        if not isinstance(accessions, list):
            raise ValueError(f"family '{family_id}' accessions must be a list")
        for accession in accessions:
            acc = str(accession).upper()
            if acc in mapping and mapping[acc] != family_id:
                raise ValueError(f"accession {acc} is assigned to multiple families")
            mapping[acc] = str(family_id)
    return mapping


def annotate_study_families(
    frame,
    *,
    accession_column: str = "accession",
    study_column: str = "study_id",
    registry_path: str | Path = DEFAULT_REGISTRY,
):
    """Attach explicit study-family IDs without inferring relationships.

    Unmapped accessions receive their own deterministic family ID. This avoids
    falsely claiming independence while still allowing ordinary candidate data
    to flow through the audit pipeline.
    """
    if accession_column not in frame.columns:
        raise ValueError(f"missing accession column: {accession_column}")
    if study_column not in frame.columns:
        raise ValueError(f"missing study column: {study_column}")

    mapping = accession_to_family(registry_path)
    out = frame.copy()
    families = []
    for accession, study_id in zip(out[accession_column], out[study_column]):
        acc = str(accession).upper()
        families.append(mapping.get(acc, f"unmapped:{study_id}"))
    out["study_family_id"] = families
    return out
