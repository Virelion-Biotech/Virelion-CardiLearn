#!/usr/bin/env python3
"""Fetch public NCBI/GEO metadata needed before a Step 19 data lock.

This script deliberately fetches metadata only. It does not download raw sequencing data,
commit expression matrices, or infer biological labels. The output is a provenance bundle
that can be inspected and reconciled before expression processing.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path

USER_AGENT = "Virelion-CardiLearn/0.3 metadata acquisition"

SOURCES = {
    "nakada_pig_2022": {"accession": "GSE185289", "kind": "geo"},
    "gao_zebrafish_2025": {"accession": "PRJNA1233465", "kind": "sra"},
    "kuppe_human_mi_2022": {"accession": "GSE217494", "kind": "geo"},
}


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_soft_samples(payload: bytes) -> list[dict[str, str]]:
    """Extract GEO sample-level fields from a SOFT family file."""
    if payload[:2] == b"\x1f\x8b":
        text = gzip.decompress(payload).decode("utf-8", errors="replace")
    else:
        text = payload.decode("utf-8", errors="replace")

    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    characteristics: list[str] = []

    for line in text.splitlines():
        if line.startswith("^SAMPLE = "):
            if current is not None:
                current["characteristics_ch1"] = " | ".join(characteristics)
                rows.append(current)
            current = {"geo_accession": line.split("=", 1)[1].strip()}
            characteristics = []
            continue
        if current is None:
            continue
        if line.startswith("!Sample_geo_accession = "):
            current["geo_accession"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_title = "):
            current["title"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_source_name_ch1 = "):
            current["source_name"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_organism_ch1 = "):
            current["organism"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_platform_id = "):
            current["platform_id"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_characteristics_ch1 = "):
            characteristics.append(line.split("=", 1)[1].strip())

    if current is not None:
        current["characteristics_ch1"] = " | ".join(characteristics)
        rows.append(current)
    return rows


def fetch_geo(accession: str, output_dir: Path) -> dict:
    series_number = re.search(r"GSE(\d+)", accession)
    if not series_number:
        raise ValueError(f"Invalid GEO series accession: {accession}")
    number = int(series_number.group(1))
    low = (number // 1000) * 1000
    soft_url = (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{low}nnn/"
        f"{accession}/soft/{accession}_family.soft.gz"
    )
    payload = fetch_bytes(soft_url)
    samples = parse_soft_samples(payload)
    raw_path = output_dir / f"{accession}_family.soft.gz"
    raw_path.write_bytes(payload)
    csv_path = output_dir / f"{accession}_samples.csv"
    fieldnames = sorted({key for row in samples for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    return {
        "kind": "geo",
        "accession": accession,
        "url": soft_url,
        "sample_count": len(samples),
        "sha256": sha256_bytes(payload),
    }


def fetch_sra(accession: str, output_dir: Path) -> dict:
    url = f"https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={accession}"
    payload = fetch_bytes(url)
    text = payload.decode("utf-8", errors="replace")
    csv_path = output_dir / f"{accession}_runinfo.csv"
    csv_path.write_text(text, encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    return {
        "kind": "sra",
        "accession": accession,
        "url": url,
        "run_count": len(rows),
        "sha256": sha256_bytes(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/ncbi_metadata"))
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCES),
        help="Fetch only selected source(s).",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = args.source or list(SOURCES)
    manifest = {"schema_version": "1.0", "sources": []}
    for source_id in selected:
        source = SOURCES[source_id]
        if source["kind"] == "geo":
            record = fetch_geo(source["accession"], args.output)
        else:
            record = fetch_sra(source["accession"], args.output)
        record["source_id"] = source_id
        manifest["sources"].append(record)
        print(json.dumps(record, sort_keys=True))
    (args.output / "acquisition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
