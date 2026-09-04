from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from cardilearn.ingestion import SparseExpression, apply_gene_map, select_variable_genes_sparse, validate_metadata


def test_sparse_expression_validates_and_selects_train_only_genes():
    X = sparse.csr_matrix(
        np.array([[1, 0, 10], [2, 0, 20], [3, 0, 30], [100, 0, 31]], dtype=np.float32)
    )
    expression = SparseExpression(X, ("a", "b", "c", "d"), ("g1", "g2", "g3"))
    selected = select_variable_genes_sparse(expression, [True, True, True, False], 1)
    assert selected.tolist() == [2]


def test_gene_map_is_conservative():
    expression = SparseExpression(
        sparse.csr_matrix(np.eye(2, dtype=np.float32)),
        ("cell1", "cell2"),
        ("pig_a", "pig_b"),
    )
    mapped = apply_gene_map(
        expression,
        pd.DataFrame({"source_gene": ["pig_a", "pig_b"], "target_gene": ["H1", "H2"]}),
    )
    assert mapped.gene_ids == ("H1", "H2")


def test_metadata_hierarchy_rejects_conflicting_sample_parent():
    metadata = pd.DataFrame(
        {
            "study_id": ["s1", "s1"],
            "subject_id": ["a", "b"],
            "sample_id": ["sample", "sample"],
            "species": ["pig", "pig"],
            "assay": ["snRNA", "snRNA"],
            "cell_type": ["CM", "CM"],
            "maturation": [1, 1],
            "injury": [0, 0],
        }
    )
    try:
        validate_metadata(metadata)
    except ValueError as exc:
        assert "sample_id" in str(exc)
    else:
        raise AssertionError("conflicting sample parent mapping must be rejected")
