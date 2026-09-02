from __future__ import annotations

import pandas as pd

from scripts.profile_geo_metadata import profile


def test_profile_is_descriptive_and_counts_raw_fields():
    frame = pd.DataFrame(
        [
            {
                "sample_id": "GSM1",
                "organism_ch1": "Mus musculus",
                "title": "P1 MI day 1",
                "raw_characteristics": {
                    "condition": ["MI"],
                    "timepoint": ["day 1"],
                    "animal": ["unknown"],
                },
            },
            {
                "sample_id": "GSM2",
                "organism_ch1": "Mus musculus",
                "title": "P1 sham day 1",
                "raw_characteristics": {
                    "condition": ["sham"],
                    "timepoint": ["day 1"],
                    "animal": ["unknown"],
                },
            },
        ]
    )

    result = profile(frame)
    assert result["samples"] == 2
    assert result["characteristic_keys"]["condition"] == 2
    assert result["raw_condition_values"]["MI"] == 1
    assert result["raw_condition_values"]["sham"] == 1
    assert result["raw_timepoint_values"]["day 1"] == 2
    assert result["organism_values"]["Mus musculus"] == 2
    assert result["duplicate_titles"] == {}
    assert result["sample_ids"] == ["GSM1", "GSM2"]
