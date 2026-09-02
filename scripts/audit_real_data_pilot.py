"""Audit the CardiLearn v0.1 real-data pilot manifest.

This script is intentionally metadata-only. It does not download or redistribute
source expression matrices. GEO materialization remains an explicit user action
through scripts/materialize_geo.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cardilearn.real_data import audit_manifest, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CardiLearn real-data pilot")
    parser.add_argument(
        "--manifest",
        default="configs/real_data_pilot_v0_1.json",
    )
    parser.add_argument(
        "--output",
        default="runs/real-data-pilot-v0.1/manifest_audit.json",
    )
    args = parser.parse_args()

    studies = load_manifest(args.manifest)
    audit = audit_manifest(studies)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))

    # Candidate manifests are allowed to be incomplete at this phase, but
    # scientifically blocking manifest errors should still fail CI/automation.
    return 1 if audit.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
