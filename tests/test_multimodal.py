import numpy as np
import pandas as pd
import pytest
from cardilearn.fusion import align_modalities, concatenate_embeddings
from cardilearn.modalities import OmicsMatrix, Waveform

def test_align_modalities_preserves_common_subjects():
    a = pd.DataFrame({"sample_id":["s1","s2"],"age":[10,20]})
    b = pd.DataFrame({"sample_id":["s2","s3"],"rate":[2,3]})
    out = align_modalities({"clinical":a, "ecg":b})
    assert out["sample_id"].tolist() == ["s2"] and "rate" in out

def test_embeddings_require_identical_order():
    with pytest.raises(ValueError):
        concatenate_embeddings({"a":(["s1","s2"], np.ones((2,2))), "b":(["s2","s1"], np.ones((2,3)))})

def test_modality_containers_validate_shape():
    x = OmicsMatrix(np.ones((2,3)), ["s1","s2"], ["g1","g2","g3"])
    assert x.log1p().matrix.shape == (2,3)
    w = Waveform(np.ones((2,1000)), 500, "s1")
    assert w.duration_s == 2
