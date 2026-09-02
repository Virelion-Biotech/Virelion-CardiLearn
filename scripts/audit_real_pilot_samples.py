"""Audit a canonical CardiLearn real-pilot sample table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cardilearn.pilot_reconciliation import audit_sample_frame, candidate_split_plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CardiLearn real pilot samples")
    parser.add_argument("--input", default="data/processed/real_pilot_samples.parquet")
    parser.add_argument("--output", default="runs/real-data-pilot-v0.1/sample_audit.json")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    audits = audit_sample_frame(frame)
    split_plan = candidate_split_plan(frame)

    payload = {
        "studies": [audit.to_dict() for audit in audits],
        "split_plan": split_plan,
        "locked": False,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))

    blocking = [
        audit
        for audit in audits
        if audit.status.startswith("blocking_")
    ]
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
