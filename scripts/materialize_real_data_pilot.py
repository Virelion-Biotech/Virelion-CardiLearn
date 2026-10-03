"""Download and normalize only the GEO family-SOFT metadata for the v0.1 pilot.

No expression matrix is downloaded. The purpose is to recover sample structure,
subject identifiers, conditions, timepoints, modality, and raw characteristics
before any biological training data are materialized.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cardilearn.geo import download_resource
from cardilearn.real_data import (
    audit_manifest,
    canonicalize_geo_samples,
    load_manifest,
    parse_geo_family_soft,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize CardiLearn pilot GEO metadata")
    parser.add_argument("--manifest", default="configs/real_data_pilot_v0_1.json")
    parser.add_argument("--cache", default="data/raw")
    parser.add_argument("--output", default="data/processed/real_pilot_samples.parquet")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    studies = load_manifest(args.manifest)
    manifest_audit = audit_manifest(studies)
    if manifest_audit.blocking:
        raise SystemExit(
            json.dumps(manifest_audit.to_dict(), indent=2, sort_keys=True)
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    resources: list[dict[str, object]] = []

    for study in studies:
        if study.source != "GEO":
            continue
        resource = download_resource(
            study.accession,
            "family_soft",
            Path(args.cache) / study.accession.upper(),
            overwrite=args.overwrite,
        )
        resources.append(resource.__dict__)
        raw = parse_geo_family_soft(resource.path)
        canonical = canonicalize_geo_samples(raw, study)
        frames.append(canonical)

        print(
            f"{study.accession}: samples={len(canonical)} "
            f"subjects={canonical['subject_id'].notna().sum()} "
            f"controlled_conditions="
            f"{(canonical['condition_status'] == 'controlled').sum()}"
        )

    if not frames:
        raise SystemExit("pilot manifest contains no GEO studies")

    combined = pd.concat(frames, ignore_index=True)
    if combined["sample_id"].eq("").any():
        raise SystemExit("one or more GEO samples have no sample_id")
    if combined["sample_id"].duplicated().any():
        raise SystemExit("duplicate sample_id detected across pilot studies")

    combined.to_parquet(output, index=False)

    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "manifest": args.manifest,
                "resources": resources,
                "rows": len(combined),
                "studies": sorted(combined["study_id"].unique().tolist()),
                "subject_ids_present": int(combined["subject_id"].notna().sum()),
                "controlled_conditions_present": int(
                    (combined["condition_status"] == "controlled").sum()
                ),
                "ready_for_lock": False,
                "reason": "sample-level metadata still requires biological reconciliation and task-specific review",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"wrote {output}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
