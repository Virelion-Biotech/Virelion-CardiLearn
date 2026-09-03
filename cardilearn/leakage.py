"""Explicit data-leakage controls for CardiLearn.

This module treats biological hierarchy, study-family independence, frozen split
assignments, and optional exact feature duplicates as separate leakage surfaces.
It intentionally cannot infer biological relationships from expression alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd


HIERARCHY_COLUMNS = ("study_family_id", "study_id", "subject_id", "sample_id")
REQUIRED_FOR_AUDIT = ("study_id", "subject_id", "sample_id", "_split")


@dataclass(frozen=True)
class LeakageFinding:
    severity: str
    code: str
    message: str
    values: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "values": list(self.values),
        }


class LeakageError(AssertionError):
    """Raised when a locked-benchmark leakage invariant is violated."""


def audit_hierarchy(
    frame: pd.DataFrame,
    *,
    split_column: str = "_split",
    hierarchy_columns: Iterable[str] = HIERARCHY_COLUMNS,
) -> list[LeakageFinding]:
    """Return findings for IDs appearing in multiple partitions or missing IDs.

    The audit is conservative: missing biological identifiers are reported as
    blocking findings rather than repaired from sample names or expression.
    """
    required = set(REQUIRED_FOR_AUDIT)
    required.add(split_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing leakage-audit columns: {missing}")

    findings: list[LeakageFinding] = []
    for column in hierarchy_columns:
        if column not in frame.columns:
            continue
        series = frame[column].replace("", pd.NA)
        missing_count = int(series.isna().sum())
        if missing_count:
            findings.append(
                LeakageFinding(
                    "blocking",
                    f"MISSING_{column.upper()}",
                    f"{missing_count} observations have no {column}; they cannot be protected by hierarchy grouping",
                )
            )
        counts = frame.assign(_hierarchy_key=series).dropna(subset=["_hierarchy_key"]).groupby(
            "_hierarchy_key", dropna=True
        )[split_column].nunique()
        leaked = counts[counts > 1].index.astype(str).tolist()
        if leaked:
            findings.append(
                LeakageFinding(
                    "blocking",
                    f"{column.upper()}_CROSS_SPLIT",
                    f"{column} appears in more than one partition",
                    tuple(leaked[:20]),
                )
            )

    duplicate_samples = int(frame["sample_id"].duplicated().sum())
    if duplicate_samples:
        findings.append(
            LeakageFinding(
                "blocking",
                "DUPLICATE_SAMPLE_ID",
                f"{duplicate_samples} duplicate sample_id rows were found",
            )
        )

    splits = set(frame[split_column].dropna().astype(str))
    expected = {"train", "validation", "test"}
    missing_splits = sorted(expected.difference(splits))
    if missing_splits:
        findings.append(
            LeakageFinding(
                "blocking",
                "MISSING_PARTITION",
                f"required partition(s) are absent: {missing_splits}",
            )
        )

    unknown_splits = sorted(splits.difference(expected))
    if unknown_splits:
        findings.append(
            LeakageFinding(
                "blocking",
                "UNKNOWN_PARTITION",
                f"unknown partition label(s): {unknown_splits}",
            )
        )

    return findings


def assert_no_leakage(frame: pd.DataFrame, *, split_column: str = "_split") -> None:
    """Raise if any blocking hierarchy/partition finding exists."""
    findings = audit_hierarchy(frame, split_column=split_column)
    blocking = [finding for finding in findings if finding.severity == "blocking"]
    if blocking:
        raise LeakageError(json.dumps([f.to_dict() for f in blocking], sort_keys=True))


def freeze_split_manifest(
    frame: pd.DataFrame,
    *,
    group_column: str = "study_family_id",
    split_column: str = "_split",
    version: str = "v0.1",
) -> dict[str, Any]:
    """Create a deterministic, hashable split manifest from group assignments.

    The manifest contains one record per independent group, never per cell, so
    it can be versioned without committing expression data.
    """
    assert_no_leakage(frame, split_column=split_column)
    if group_column not in frame.columns:
        raise ValueError(f"missing group column: {group_column}")

    groups = frame[[group_column, split_column]].drop_duplicates().sort_values(group_column)
    if groups[group_column].duplicated().any():
        raise LeakageError(f"group {group_column} has multiple split assignments")

    records = [
        {"group": str(row[group_column]), "split": str(row[split_column])}
        for _, row in groups.iterrows()
    ]
    payload = {
        "version": version,
        "group_column": group_column,
        "split_column": split_column,
        "records": records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    payload["locked"] = False
    payload["policy"] = {
        "test_protected": True,
        "assignment_unit": group_column,
        "cells_never_split_directly": True,
    }
    return payload


def verify_frozen_split_manifest(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    """Verify that current group assignments exactly match a saved manifest."""
    group_column = str(manifest.get("group_column", ""))
    split_column = str(manifest.get("split_column", "_split"))
    records = manifest.get("records")
    expected_hash = manifest.get("sha256")
    if not group_column or not isinstance(records, list) or not expected_hash:
        raise ValueError("invalid frozen split manifest")
    if group_column not in frame.columns or split_column not in frame.columns:
        raise ValueError("frame is missing frozen split columns")

    current = frame[[group_column, split_column]].drop_duplicates().sort_values(group_column)
    current_records = [
        {"group": str(row[group_column]), "split": str(row[split_column])}
        for _, row in current.iterrows()
    ]
    if current_records != records:
        raise LeakageError("current split assignments differ from frozen manifest")

    payload = {
        "version": manifest.get("version"),
        "group_column": group_column,
        "split_column": split_column,
        "records": records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    actual_hash = hashlib.sha256(canonical).hexdigest()
    if actual_hash != expected_hash:
        raise LeakageError("frozen split manifest hash does not match its contents")


def exact_cross_split_feature_collisions(
    X: np.ndarray,
    splits: Iterable[str],
) -> dict[str, Any]:
    """Find exact duplicated rows occurring across different partitions.

    This is an adversarial check, not a blanket ban on duplicate cells: exact
    duplicates within a biological sample can be legitimate in sparse counts.
    Cross-partition duplicates are the suspicious case.
    """
    X = np.asarray(X)
    split_values = np.asarray(list(splits), dtype=object)
    if X.ndim != 2 or len(split_values) != len(X):
        raise ValueError("X and splits have incompatible shapes")

    owners: dict[str, set[str]] = {}
    for row, split in zip(X, split_values):
        digest = hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        owners.setdefault(digest, set()).add(str(split))

    collisions = sorted(digest for digest, split_set in owners.items() if len(split_set) > 1)
    return {
        "unique_row_hashes": len(owners),
        "cross_split_collision_count": len(collisions),
        "collision_hashes": collisions[:100],
        "status": "blocking" if collisions else "clear",
    }
