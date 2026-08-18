import numpy as np
import pandas as pd
import pytest
from cardilearn.calibration import expected_calibration_error
from cardilearn.reproducibility import dataframe_fingerprint
from cardilearn.validation import assert_no_group_overlap, validate_dataset
from cardilearn.temporal import forward_split

def test_dataset_validation_flags_missing_target_and_duplicates():
    df = pd.DataFrame({"sample_id":["a","a","b"],"group_id":[1,1,2],"target":[1,np.nan,0],"x":[1,2,3]})
    r = validate_dataset(df, target="target", id_column="sample_id", group_column="group_id")
    assert not r.ok and r.duplicate_ids == 1 and r.target_missing == 1

def test_group_overlap_is_rejected():
    train = pd.DataFrame({"group":[1,2]}); test = pd.DataFrame({"group":[2,3]})
    with pytest.raises(AssertionError): assert_no_group_overlap(train, test, "group")

def test_forward_split_is_chronological():
    df = pd.DataFrame({"time":[3,1,4,2,5],"x":range(5)})
    train, val, test = forward_split(df, "time", validation_fraction=.2, test_fraction=.2)
    assert train["time"].max() < val["time"].min() < test["time"].min()

def test_fingerprint_changes_with_data():
    a = pd.DataFrame({"x":[1,2]}); b = pd.DataFrame({"x":[1,3]})
    assert dataframe_fingerprint(a) != dataframe_fingerprint(b)

def test_ece_is_bounded():
    ece = expected_calibration_error(np.array([0,1,1,0]), np.array([.1,.9,.8,.2]))
    assert 0 <= ece <= 1
