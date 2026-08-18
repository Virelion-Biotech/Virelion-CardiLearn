import pandas as pd
import pytest

from cardilearn.cardiobench import BenchmarkDefinition, load_definition, holdout_by_column, validate_definition_against_frame


def test_definition_requires_contract_fields():
    with pytest.raises(ValueError, match="missing fields"):
        BenchmarkDefinition.from_mapping({"benchmark_id": "x"})


def test_definition_validates_against_frame():
    definition = BenchmarkDefinition(
        benchmark_id="mi",
        version="1",
        task="classification",
        target="target",
        split_policy="animal",
        group_key="animal_id",
    )
    frame = pd.DataFrame({"target": [0, 1], "animal_id": ["a", "b"], "x": [1, 2]})
    validate_definition_against_frame(definition, frame)


def test_holdout_by_column_keeps_held_out_level_out_of_development():
    frame = pd.DataFrame(
        {
            "study": ["A"] * 12 + ["B"] * 6,
            "group_id": [f"g{i}" for i in range(18)],
            "target": [0, 1] * 9,
            "x": range(18),
        }
    )
    splits = holdout_by_column(frame, "study", "B", group_column="group_id")
    assert set(splits.test) == set(range(12, 18))
    assert set(splits.train).isdisjoint(set(splits.test))
    assert set(splits.validation).isdisjoint(set(splits.test))
