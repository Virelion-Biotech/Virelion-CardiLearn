"""Subject-aligned multimodal fusion primitives."""
from __future__ import annotations
import numpy as np
import pandas as pd

def align_modalities(modalities: dict[str, pd.DataFrame], id_column: str = "sample_id") -> pd.DataFrame:
    if not modalities: raise ValueError("at least one modality is required")
    names = list(modalities)
    base = modalities[names[0]].copy()
    if id_column not in base: raise ValueError(f"missing {id_column} in {names[0]}")
    base = base.set_index(id_column)
    for name in names[1:]:
        frame = modalities[name].copy()
        if id_column not in frame: raise ValueError(f"missing {id_column} in {name}")
        frame = frame.set_index(id_column)
        overlap = set(base.columns) & set(frame.columns)
        frame = frame.rename(columns={c: f"{name}__{c}" for c in overlap})
        base = base.join(frame, how="inner", validate="one_to_one")
    if base.empty: raise ValueError("no common sample IDs across modalities")
    return base.reset_index()

def concatenate_embeddings(embeddings: dict[str, tuple[list[str], np.ndarray]]) -> tuple[list[str], np.ndarray]:
    ids = None; blocks=[]
    for name, (current_ids, matrix) in embeddings.items():
        if matrix.ndim != 2: raise ValueError(f"{name} embedding must be 2D")
        if ids is None: ids=list(current_ids)
        elif ids != list(current_ids): raise ValueError("embedding IDs are not identically ordered")
        blocks.append(matrix)
    return ids, np.concatenate(blocks, axis=1)
