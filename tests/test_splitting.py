import numpy as np
import pandas as pd
import pytest

from cardilearn.config import SplitConfig
from cardilearn.splitting import split_frame


def test_group_split_has_no_group_leakage():
    frame = pd.DataFrame({"x": np.arange(30), "target": [0, 1] * 15})
    groups = pd.Series([f"g{i // 3}" for i in range(30)])
    splits = split_frame(frame, SplitConfig(), target=frame["target"], groups=groups)

    for left, right in ((splits.train, splits.validation), (splits.train, splits.test), (splits.validation, splits.test)):
        assert set(groups.iloc[left]).isdisjoint(set(groups.iloc[right]))
    splits.assert_disjoint()


def test_group_split_rejects_too_few_groups():
    frame = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
    groups = pd.Series(["a", "a", "b"])
    with pytest.raises(ValueError, match="at least 3 groups"):
        split_frame(frame, SplitConfig(), groups=groups)
