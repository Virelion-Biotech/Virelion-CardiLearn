from pathlib import Path

from scripts.reconcile_geo_samples import reconcile


def test_reconcile_preserves_unresolved_labels(tmp_path: Path):
    soft = tmp_path / "GSETEST_family.soft"
    soft.write_text(
        "\n".join(
            [
                "^SAMPLE = GSM1",
                "!Sample_title = animal one sham",
                "!Sample_geo_accession = GSM1",
                "!Sample_organism_ch1 = Mus musculus",
                "!Sample_characteristics_ch1 = animal_id: A1",
                "!Sample_characteristics_ch1 = condition: sham",
                "!Sample_characteristics_ch1 = timepoint: P1",
                "!Sample_characteristics_ch1 = tissue: heart",
                "^SAMPLE = GSM2",
                "!Sample_title = animal two unknown",
                "!Sample_geo_accession = GSM2",
                "!Sample_organism_ch1 = Mus musculus",
                "!Sample_characteristics_ch1 = animal_id: A2",
                "!Sample_characteristics_ch1 = condition: mystery",
            ]
        ),
        encoding="utf-8",
    )

    frame = reconcile(
        soft,
        "test_study",
        "GSETEST",
        "Mus musculus",
        "scRNA-seq",
    )

    assert len(frame) == 2
    assert frame.loc[0, "subject_id"] == "A1"
    assert frame.loc[0, "condition"] == "reference"
    assert frame.loc[0, "condition_status"] == "controlled"
    assert frame.loc[0, "subject_confidence"] == "source_explicit_unreviewed"
    assert frame.loc[1, "condition_status"] == "unresolved"
    assert "condition_unresolved" in frame.loc[1, "review_issues"]
    assert frame["maturation"].eq("").all()
    assert frame["injury"].eq("").all()
