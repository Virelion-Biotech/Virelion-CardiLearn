"""Create a cryptographically identified CardiLearn real-data lock.

The command is intentionally strict. It refuses to lock candidate metadata, unresolved
biology, duplicate observation IDs, hierarchy conflicts, or mismatched expression /
metadata identities. It records fingerprints and a frozen study-level split manifest;
it does not train a model.
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


def build_lock(metadata: pd.DataFrame, expression: SparseExpression) -> dict[str, object]:
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
    if metadata["sample_id"].duplicated().any():
        # Sample IDs may repeat across cells; uniqueness is checked only as a mapping.
        mapping = metadata.groupby("sample_id")["subject_id"].nunique()
        if (mapping > 1).any():
            raise ValueError("a sample maps to multiple subjects")
    family_frame = annotate_study_families(metadata)
    studies = sorted(family_frame["study_id"].astype(str).unique().tolist())
    families = sorted(family_frame["study_family_id"].astype(str).unique().tolist())
    if len(families) < 6:
        raise ValueError("at least six independent study families are required for the locked multi-study protocol")

    # Deterministic study-level split: 60/20/20 by sorted family IDs.
    n = len(families)
    n_train = max(1, int(np.floor(n * 0.6)))
    n_val = max(1, int(np.floor(n * 0.2)))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    train_families = families[:n_train]
    val_families = families[n_train : n_train + n_val]
    test_families = families[n_train + n_val :]
    split = {
        "unit": "study_family_id",
        "train": train_families,
        "validation": val_families,
        "test": test_families,
    }
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
        "split_fingerprint": fingerprint_ids(train_families + val_families + test_families),
        "guardrails": {
            "study_level_split": True,
            "subject_verified": True,
            "conditions_controlled": True,
            "train_only_preprocessing_required": True,
            "test_locked": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a CardiLearn real-data lock")
    parser.add_argument("--expression", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expression = load_npz_expression(args.expression)
    metadata = pd.read_parquet(args.metadata)
    lock = build_lock(metadata, expression)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
