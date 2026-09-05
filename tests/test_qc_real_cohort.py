import numpy as np
import pandas as pd
from scipy import sparse

from scripts.qc_real_cohort import qc_report


def test_qc_report_detects_hierarchy_and_zero_cells():
    X = sparse.csr_matrix(np.array([[1, 0, 2], [0, 0, 0], [3, 1, 0]], dtype=np.float32))
    metadata = pd.DataFrame(
        {
            "observation_id": ["c1", "c2", "c3"],
            "study_id": ["s1", "s1", "s2"],
            "subject_id": ["a1", "a1", "a2"],
            "sample_id": ["m1", "m1", "m2"],
        }
    )
    report = qc_report(X, metadata)
    assert report["n_observations"] == 3
    assert report["zero_observations"] == 1
    assert report["n_samples"] == 2
    assert report["ambiguous_sample_to_subject"] == []


def test_qc_report_rejects_row_mismatch():
    X = sparse.csr_matrix(np.ones((2, 3), dtype=np.float32))
    metadata = pd.DataFrame(
        {"study_id": ["s1"], "subject_id": ["a1"], "sample_id": ["m1"]}
    )
    try:
        qc_report(X, metadata)
    except ValueError as exc:
        assert "metadata rows" in str(exc)
    else:
        raise AssertionError("expected row mismatch to fail")
