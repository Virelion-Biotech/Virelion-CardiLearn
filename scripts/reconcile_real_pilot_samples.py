"""Apply conservative study-specific reconciliation to the real pilot table.

This script does not decide unresolved biology automatically. It adds provenance,
confidence, and candidate biological-unit fields that are safe for review. A row
with an unresolved biological parent remains ineligible for a locked split.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


GSE185289_ID = re.compile(r"(?<!\d)(\d{4,5})(?=[A-Za-z_]|$)")
GSE217494_PAIRED = re.compile(r"^(?:GEX|Ab)_sample(\d+)$", re.IGNORECASE)


def _first_nonempty(*values: object) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def reconcile_gse185289(row: pd.Series) -> tuple[str, str, str]:
    """Return candidate animal ID, provenance, and confidence."""
    source_name = _first_nonempty(row.get("source_name"), row.get("sample_id"))
    title = _first_nonempty(row.get("title"), row.get("sample_id"))
    match = GSE185289_ID.search(source_name) or GSE185289_ID.search(title)
    if not match:
        return "", "unresolved", "unresolved"
    return f"pig_{match.group(1)}", "geo_source_name_or_title", "high_candidate"


def reconcile_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["subject_id_candidate"] = ""
    out["subject_provenance"] = "unresolved"
    out["subject_confidence"] = "unresolved"
    out["biological_unit_status"] = "unresolved"
    out["paired_measurement_id"] = ""

    for idx, row in out.iterrows():
        study_id = str(row.get("study_id", ""))
        sample_id = str(row.get("sample_id", ""))

        if study_id == "gse185289_pig_regeneration":
            subject, provenance, confidence = reconcile_gse185289(row)
            out.at[idx, "subject_id_candidate"] = subject
            out.at[idx, "subject_provenance"] = provenance
            out.at[idx, "subject_confidence"] = confidence
            out.at[idx, "biological_unit_status"] = (
                "candidate_animal" if subject else "unresolved"
            )

        elif study_id == "gse217494_human_mi_cite":
            match = GSE217494_PAIRED.match(sample_id)
            if match:
                out.at[idx, "paired_measurement_id"] = f"human_heart_{match.group(1)}"
                out.at[idx, "subject_provenance"] = "paired_GEX_Ab_sample_index"
                out.at[idx, "subject_confidence"] = "high_candidate"
                out.at[idx, "biological_unit_status"] = "candidate_human_heart"

        else:
            # Deliberately unresolved: titles alone do not establish the
            # biological parent for the remaining candidate studies.
            pass

    out["locked_eligible"] = (
        out["subject_confidence"].eq("verified")
        & out["condition_status"].eq("controlled")
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile CardiLearn real-pilot samples")
    parser.add_argument("--input", default="data/processed/real_pilot_samples.parquet")
    parser.add_argument("--output", default="data/processed/real_pilot_samples_reconciled.parquet")
    parser.add_argument("--report", default="runs/real-data-pilot-v0.1/reconciliation_report.json")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    out = reconcile_frame(frame)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    report = {
        "rows": int(len(out)),
        "candidate_subject_rows": int((out["subject_confidence"] == "high_candidate").sum()),
        "verified_subject_rows": int((out["subject_confidence"] == "verified").sum()),
        "unresolved_subject_rows": int((out["subject_confidence"] == "unresolved").sum()),
        "locked_eligible_rows": int(out["locked_eligible"].sum()),
        "locked": False,
        "reason": "candidate mappings require explicit source cross-check before verification and split lock",
        "studies": {},
    }
    for study_id, group in out.groupby("study_id", sort=True):
        report["studies"][study_id] = {
            "rows": int(len(group)),
            "candidate_subject_rows": int((group["subject_confidence"] == "high_candidate").sum()),
            "verified_subject_rows": int((group["subject_confidence"] == "verified").sum()),
            "unresolved_subject_rows": int((group["subject_confidence"] == "unresolved").sum()),
            "controlled_condition_rows": int((group["condition_status"] == "controlled").sum()),
        }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
