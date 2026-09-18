"""Validated pairing and subject-aligned multimodal fusion primitives.

True multimodal alignment requires one-to-one biological identifiers. Weak
correspondence and synthetic pairing are represented explicitly rather than
being silently promoted to true pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairingAudit:
    id_column: str
    modality_counts: dict[str, int]
    common_count: int
    duplicate_counts: dict[str, int]
    status: str
    notes: tuple[str, ...] = ()


def audit_pairing(
    modalities: dict[str, pd.DataFrame],
    *,
    id_column: str = "sample_id",
) -> PairingAudit:
    if not modalities:
        raise ValueError("at least one modality is required")
    duplicate_counts: dict[str, int] = {}
    counts: dict[str, int] = {}
    id_sets: list[set[str]] = []
    for name, frame in modalities.items():
        if id_column not in frame:
            raise ValueError(f"missing {id_column} in {name}")
        ids = frame[id_column].astype(str)
        duplicate_counts[name] = int(ids.duplicated().sum())
        counts[name] = int(ids.nunique())
        id_sets.append(set(ids))
    common = set.intersection(*id_sets)
    if any(count > 0 for count in duplicate_counts.values()):
        status = "invalid_duplicate_ids"
    elif not common:
        status = "no_pairs"
    else:
        status = "paired"
    return PairingAudit(
        id_column=id_column,
        modality_counts=counts,
        common_count=len(common),
        duplicate_counts=duplicate_counts,
        status=status,
    )


def align_modalities(
    modalities: dict[str, pd.DataFrame],
    id_column: str = "sample_id",
) -> pd.DataFrame:
    audit = audit_pairing(modalities, id_column=id_column)
    if audit.status == "invalid_duplicate_ids":
        raise ValueError(f"duplicate biological IDs prevent one-to-one pairing: {audit.duplicate_counts}")
    if audit.common_count == 0:
        raise ValueError("no common biological IDs across modalities")
    names = list(modalities)
    base = modalities[names[0]].copy().set_index(id_column)
    for name in names[1:]:
        frame = modalities[name].copy().set_index(id_column)
        overlap = set(base.columns) & set(frame.columns)
        frame = frame.rename(columns={c: f"{name}__{c}" for c in overlap})
        base = base.join(frame, how="inner", validate="one_to_one")
    return base.reset_index()


def paired_id_intersection(
    modalities: dict[str, pd.DataFrame],
    *,
    id_column: str = "sample_id",
) -> tuple[str, ...]:
    audit = audit_pairing(modalities, id_column=id_column)
    if audit.status != "paired":
        raise ValueError(f"modalities are not valid true pairs: {audit.status}")
    common = set.intersection(*(set(frame[id_column].astype(str)) for frame in modalities.values()))
    return tuple(sorted(common))


def concatenate_embeddings(
    embeddings: dict[str, tuple[list[str], np.ndarray]],
) -> tuple[list[str], np.ndarray]:
    if not embeddings:
        raise ValueError("at least one embedding is required")
    ids: list[str] | None = None
    blocks = []
    for name, (current_ids, matrix) in embeddings.items():
        current_ids = [str(value) for value in current_ids]
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != len(current_ids):
            raise ValueError(f"{name} embedding must be [n_ids, dim] with matching IDs")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} embedding contains non-finite values")
        if ids is None:
            ids = current_ids
        elif ids != current_ids:
            raise ValueError("embedding IDs are not identically ordered")
        blocks.append(matrix)
    assert ids is not None
    return ids, np.concatenate(blocks, axis=1)
