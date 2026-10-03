from __future__ import annotations

import pandas as pd

from scripts.reconcile_real_pilot_samples import reconcile_frame


def test_pig_source_name_generates_candidate_animal_without_verifying_it():
    frame = pd.DataFrame(
        [
            {
                "study_id": "gse185289_pig_regeneration",
                "sample_id": "GSM1",
                "source_name": "8064AZ",
                "title": "AR1_MI28_P30_8064AZ",
                "condition": "",
                "condition_status": "unresolved",
            }
        ]
    )
    out = reconcile_frame(frame)
    assert out.loc[0, "subject_id_candidate"] == "pig_8064"
    assert out.loc[0, "subject_confidence"] == "high_candidate"
    assert not bool(out.loc[0, "locked_eligible"])


def test_human_gex_and_antibody_records_share_candidate_heart_identity():
    frame = pd.DataFrame(
        [
            {
                "study_id": "gse217494_human_mi_cite",
                "sample_id": "GEX_sample12",
                "condition": "reference",
                "condition_status": "controlled",
            },
            {
                "study_id": "gse217494_human_mi_cite",
                "sample_id": "Ab_sample12",
                "condition": "reference",
                "condition_status": "controlled",
            },
        ]
    )
    out = reconcile_frame(frame)
    assert set(out["paired_measurement_id"]) == {"human_heart_12"}
    assert set(out["subject_confidence"]) == {"high_candidate"}
    assert not out["locked_eligible"].any()


def test_unmapped_study_remains_unresolved():
    frame = pd.DataFrame(
        [
            {
                "study_id": "gse130699_mouse_neonatal_injury",
                "sample_id": "GSM3747856",
                "condition": "myocardial_injury",
                "condition_status": "controlled",
            }
        ]
    )
    out = reconcile_frame(frame)
    assert out.loc[0, "subject_id_candidate"] == ""
    assert out.loc[0, "subject_confidence"] == "unresolved"
    assert not bool(out.loc[0, "locked_eligible"])
