"""Create a cryptographically identified CardiLearn real-data lock.

The command is intentionally strict. It refuses to lock candidate metadata, unresolved
biology, duplicate observation IDs, hierarchy conflicts, or mismatched expression /
metadata identities. A scientifically reviewed split manifest must be supplied;
this script never invents a convenient split from sorted IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cardilearn.ingestion import SparseExpression, validate_metadata
from cardilearn.reproducibility import dataframe_fingerprint, fingerprint_ids
from cardilearn.study_families import annotate_study_families


def _matrix_fingerprint(expression: SparseExpression) -> str:
    expression.validate()
    digest = hashlib.sha256()
    digest.update(np.asarray(expression.X.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(expression.X.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(expression.X.data, dtype=np.float32).tobytes())
    digest.update(json.dumps(list(expression.observation_ids), separators=(",", ":")).encode())
    digest.update(json.dumps(list(expression.gene_ids), separators=(",", ":")).encode())
    return digest.hexdigest()


def load_npz_expression(path: str | Path) -> SparseExpression:
    """Load a train-ready NPZ produced by the curated ingestion pipeline."""
    from scipy import sparse

    payload = np.load(path, allow_pickle=False)
    required = {"data", "indices", "indptr", "shape", "observation_ids", "gene_ids"}
    missing = required.difference(payload.files)
    if missing:
        raise ValueError(f"expression NPZ missing fields: {sorted(missing)}")
    matrix = sparse.csr_matrix(
        (payload["data"], payload["indices"], payload["indptr"]),
        shape=tuple(payload["shape"].tolist()),
    )
    return SparseExpression(
        matrix,
        tuple(payload["observation_ids"].astype(str).tolist()),
        tuple(payload["gene_ids"].astype(str).tolist()),
    )


def _validate_split(split: dict[str, object], families: set[str]) -> None:
    if split.get("unit") != "study_family_id":
        raise ValueError("locked split must use study_family_id as its unit")
    partitions = {name: split.get(name) for name in ("train", "validation", "test")}
    if any(not isinstance(value, list) or not value for value in partitions.values()):
        raise ValueError("train, validation, and test must each contain non-empty family lists")
    listed = [str(item) for value in partitions.values() for item in value]
    if len(listed) != len(set(listed)):
        raise ValueError("study families may occur in only one split")
    if set(listed) != families:
        raise ValueError("split must account for every independent study family exactly once")


def build_lock(
    metadata: pd.DataFrame,
    expression: SparseExpression,
    split: dict[str, object],
) -> dict[str, object]:
    """Validate biological readiness and return an immutable lock payload."""
    validate_metadata(metadata)
    if "observation_id" not in metadata.columns:
        raise ValueError("metadata must contain observation_id matching expression rows")
    if metadata["observation_id"].duplicated().any():
        raise ValueError("metadata contains duplicate observation_id values")
    if set(metadata["observation_id"].astype(str)) != set(expression.observation_ids):
        raise ValueError("expression and metadata observation IDs do not match exactly")
    if "subject_confidence" not in metadata.columns:
        raise ValueError("subject_confidence is required; candidate subjects cannot be locked")
    if not metadata["subject_confidence"].eq("verified").all():
        raise ValueError("all subjects must be explicitly verified before data lock")
    if "condition_status" not in metadata.columns or not metadata["condition_status"].eq("controlled").all():
        raise ValueError("all conditions must be explicitly controlled before data lock")
    mapping = metadata.groupby("sample_id")["subject_id"].nunique()
    if (mapping > 1).any():
        raise ValueError("a sample maps to multiple subjects")

    family_frame = annotate_study_families(metadata)
    families = set(family_frame["study_family_id"].astype(str).unique())
    if len(families) < 6:
        raise ValueError("at least six independent study families are required for the locked multi-study protocol")
    _validate_split(split, families)

    studies = sorted(family_frame["study_id"].astype(str).unique().tolist())
    return {
        "schema_version": "1.0",
        "locked": True,
        "study_count": len(studies),
        "independent_family_count": len(families),
        "observation_count": len(expression.observation_ids),
        "gene_count": len(expression.gene_ids),
        "expression_fingerprint": _matrix_fingerprint(expression),
        "metadata_fingerprint": dataframe_fingerprint(metadata.sort_values("observation_id").reset_index(drop=True)),
        "observation_id_fingerprint": fingerprint_ids(sorted(expression.observation_ids)),
        "gene_id_fingerprint": fingerprint_ids(list(expression.gene_ids)),
        "split": split,
        "split_fingerprint": fingerprint_ids(
            [str(x) for x in split["train"]]
            + [str(x) for x in split["validation"]]
            + [str(x) for x in split["test"]]
        ),
        "guardrails": {
            "study_family_level_split": True,
            "subject_verified": True,
            "conditions_controlled": True,
            "train_only_preprocessing_required": True,
            "test_locked": True,
            "split_reviewed_externally": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a CardiLearn real-data lock")
    parser.add_argument("--expression", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--split", required=True, help="Reviewed JSON split manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expression = load_npz_expression(args.expression)
    metadata = pd.read_parquet(args.metadata)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    lock = build_lock(metadata, expression, split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
