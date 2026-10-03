"""Stable prediction export for CardiEval and downstream consumers."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

def export_predictions(sample_ids, y_true, y_pred, *, scores=None, path: str | Path = "predictions.json") -> Path:
    rows = []
    for i, sid in enumerate(sample_ids):
        row = {"sample_id": str(sid), "y_true": y_true[i] if y_true is not None else None, "y_pred": y_pred[i]}
        if scores is not None: row["score"] = scores[i]
        rows.append(row)
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv": pd.DataFrame(rows).to_csv(p, index=False)
    else: p.write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    return p
