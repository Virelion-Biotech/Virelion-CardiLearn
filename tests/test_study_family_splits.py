from __future__ import annotations

import pandas as pd
import pytest

from cardilearn.prototype.splits import assign_split, assert_no_hierarchy_leakage, study_split


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"study_id": "GSE130699", "study_family_id": "family_a", "subject_id": "a1", "sample_id": "a1s"},
            {"study_id": "GSE153480", "study_family_id": "family_a", "subject_id": "a2", "sample_id": "a2s"},
            {"study_id": "GSE185289", "study_family_id": "family_b", "subject_id": "b1", "sample_id": "b1s"},
            {"study_id": "GSE217494", "study_family_id": "family_c", "subject_id": "c1", "sample_id": "c1s"},
        ]
    )


def test_family_group_split_keeps_linked_studies_together():
    frame = _frame()
    splits = study_split(frame, seed=7, train_fraction=0.5, val_fraction=0.25, group_column="study_family_id")
    assert splits["group_column"] == "study_family_id"

    assigned = assign_split(frame, splits, group_column="study_family_id")
    assert assigned.groupby("study_family_id")["_split"].nunique().max() == 1
    assert assigned.loc[assigned["study_id"].eq("GSE130699"), "_split"].iloc[0] == assigned.loc[assigned["study_id"].eq("GSE153480"), "_split"].iloc[0]
    assert_no_hierarchy_leakage(assigned)


def test_family_group_requires_three_independent_groups():
    frame = _frame()
    small = frame[frame["study_family_id"].isin(["family_a", "family_b"])]
    with pytest.raises(ValueError, match="at least three independent groups"):
        study_split(small, group_column="study_family_id")
