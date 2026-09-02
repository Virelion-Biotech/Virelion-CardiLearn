"""Profile recovered GEO sample metadata without making biological assignments.

The profiler is intentionally descriptive. It surfaces characteristic keys,
raw condition/timepoint values, and sample-title patterns so that study-specific
reconciliation can be reviewed before canonical IDs are assigned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

from cardilearn.geo import download_resource
from cardilearn.real_data import load_manifest, parse_geo_family_soft


def _flatten(values: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for items in values.values():
        out.extend(items)
    return out


def profile(frame) -> dict[str, object]:
    characteristic_keys: Counter[str] = Counter()
    condition_values: Counter[str] = Counter()
    timepoint_values: Counter[str] = Counter()
    organism_values: Counter[str] = Counter()
    title_values: Counter[str] = Counter()

    for _, row in frame.iterrows():
        chars = row.get("raw_characteristics") or {}
        characteristic_keys.update(chars.keys())
        condition_values.update(_flatten({"condition": chars.get("condition", [])}))
        condition_values.update(_flatten({"group": chars.get("group", [])}))
        timepoint_values.update(_flatten({"timepoint": chars.get("timepoint", [])}))
        timepoint_values.update(_flatten({"collection_timepoint": chars.get("collection_timepoint", [])}))
        if row.get("organism_ch1"):
            organism_values[str(row["organism_ch1"])] += 1
        if row.get("title"):
            title_values[str(row["title"])] += 1

    return {
        "samples": int(len(frame)),
        "characteristic_keys": dict(characteristic_keys.most_common()),
        "raw_condition_values": dict(condition_values.most_common()),
        "raw_timepoint_values": dict(timepoint_values.most_common()),
        "organism_values": dict(organism_values.most_common()),
        "duplicate_titles": {
            value: count for value, count in title_values.items() if count > 1
        },
        "sample_ids": [str(x) for x in frame["sample_id"].tolist()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile CardiLearn GEO metadata")
    parser.add_argument("--manifest", default="configs/real_data_pilot_v0_1.json")
    parser.add_argument("--cache", default="data/raw")
    parser.add_argument("--output", default="runs/real-data-pilot-v0.1/metadata_profile.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    studies = load_manifest(args.manifest)
    reports: dict[str, object] = {}
    for study in studies:
        if study.source != "GEO":
            continue
        resource = download_resource(
            study.accession,
            "family_soft",
            Path(args.cache) / study.accession.upper(),
            overwrite=args.overwrite,
        )
        frame = parse_geo_family_soft(resource.path)
        reports[study.study_id] = {
            "accession": study.accession,
            "resource": resource.__dict__,
            "profile": profile(frame),
            "descriptive_only": True,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(reports, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(reports, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
