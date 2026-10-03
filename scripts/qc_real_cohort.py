"""Structural QC for assembled CardiLearn real-data cohorts.

This script deliberately performs no biological inference and never approves a
cohort. It reports deterministic structural and distributional checks that must
be reviewed before a real-data lock.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def load_expression(path: str | Path) -> sparse.csr_matrix:
    X = sparse.load_npz(path).tocsr()
    if X.ndim != 2:
        raise ValueError("expression matrix must be two-dimensional")
    if not np.isfinite(X.data).all():
        raise ValueError("expression contains non-finite values")
    return X


def qc_report(X: sparse.csr_matrix, metadata: pd.DataFrame) -> dict[str, object]:
    if X.shape[0] != len(metadata):
        raise ValueError("expression rows and metadata rows do not match")
    if metadata.empty:
        raise ValueError("metadata is empty")
    required = ("study_id", "subject_id", "sample_id")
    missing = [c for c in required if c not in metadata.columns]
    if missing:
        raise ValueError(f"missing required hierarchy columns: {missing}")

    obs_ids = metadata.get("observation_id")
    duplicate_observations = int(obs_ids.duplicated().sum()) if obs_ids is not None else None
    counts = np.asarray(X.sum(axis=1)).ravel()
    detected = np.asarray((X > 0).sum(axis=1)).ravel()
    sample_sizes = metadata.groupby("sample_id", dropna=False).size()
    subject_per_sample = metadata.groupby("sample_id", dropna=False)["subject_id"].nunique()
    study_per_subject = metadata.groupby("subject_id", dropna=False)["study_id"].nunique()

    return {
        "n_observations": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "nonzero_fraction": float(X.nnz / max(1, X.shape[0] * X.shape[1])),
        "zero_observations": int((counts == 0).sum()),
        "library_size": {
            "min": float(np.min(counts)),
            "median": float(np.median(counts)),
            "max": float(np.max(counts)),
        },
        "detected_features": {
            "min": int(np.min(detected)),
            "median": float(np.median(detected)),
            "max": int(np.max(detected)),
        },
        "n_studies": int(metadata["study_id"].nunique()),
        "n_subjects": int(metadata["subject_id"].nunique()),
        "n_samples": int(metadata["sample_id"].nunique()),
        "sample_cell_counts": {str(k): int(v) for k, v in sample_sizes.items()},
        "ambiguous_sample_to_subject": [str(k) for k, v in subject_per_sample.items() if v > 1],
        "ambiguous_subject_to_study": [str(k) for k, v in study_per_subject.items() if v > 1],
        "duplicate_observations": duplicate_observations,
        "required_fields_nonnull": bool(metadata[list(required)].notna().all().all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    X = load_expression(args.expression)
    metadata = pd.read_parquet(args.metadata)
    report = qc_report(X, metadata)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
