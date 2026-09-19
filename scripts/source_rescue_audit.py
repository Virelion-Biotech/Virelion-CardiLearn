"""Audit alternative public sources for the locked real-data benchmark.

The auditor is intentionally fail-closed. It discovers candidate count sources but never
promotes a transformed/derived matrix to "raw integer counts" merely because its values
are integers.

Sources:
- NCBI GEO supplementary metadata/files
- ENA/SRA run metadata for the exact GSM -> SRX -> SRR mapping
- EMBL-EBI BioStudies / ArrayExpress and Expression Atlas evidence
- ARCHS4 H5 metadata when a local H5 is supplied
- recount3 SRA-project availability via its documented metadata mirrors

The output is an evidence report. Human approval is still required before a source is
added to Step 3's EXTERNAL_COUNT_SOURCES.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import ssl
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

import h5py
import pandas as pd


USER_AGENT = "Virelion-CardiLearn/source-rescue-audit/1.0"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ENA_API = "https://www.ebi.ac.uk/ena/portal/api/filereport"
BIOSTUDIES_SEARCH = "https://www.ebi.ac.uk/biostudies/api/v1/arrayexpress/search"
BIOSTUDIES_STUDY = "https://www.ebi.ac.uk/biostudies/api/v1/studies"
ATLAS_EXPERIMENT = "https://www.ebi.ac.uk/gxa/experiments/"
RECOUNT3_MIRRORS = (
    "https://recount-opendata.s3.amazonaws.com/recount3/release",
    "https://duffel.rail.bio/recount3",
)

ACCESSION_RE = re.compile(r"\b(?:GSE|GSM|SRP|SRX|SRR|PRJNA|ERP|DRP)\d+\b", re.I)
GSM_RE = re.compile(r"\bGSM\d+\b", re.I)
SRX_RE = re.compile(r"\bSRX\d+\b", re.I)
SRR_RE = re.compile(r"\bSRR\d+\b", re.I)
PROJECT_RE = re.compile(r"\b(?:SRP|ERP|DRP)\d+\b", re.I)
RAW_TOKENS = ("raw", "count", "counts", "readcount", "read_count", "gene_count", "gene_counts")
REJECT_TOKENS = ("fpkm", "tpm", "cpm", "rpkm", "normalized", "normalised", "log2", "log1p", "vst")
STRICT_SOURCE_TYPES = {
    "geo_submitter_raw_counts",
    "expression_atlas_raw_counts",
}
DERIVED_SOURCE_TYPES = {
    "recount3_derived_read_counts",
}
REJECT_SOURCE_TYPES = {
    "archs4_kallisto_rounded",
    "geo_normalized_or_estimated",
}


@dataclass
class SourceEvidence:
    source_type: str
    status: str
    accession: str
    sample_ids: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    candidate_files: list[str] = field(default_factory=list)
    exact_sample_coverage: int = 0
    exact_sample_expected: int = 0
    run_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


def http_text(url: str, *, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    context = ssl.create_default_context()
    with urlopen(request, timeout=timeout, context=context) as response:
        return response.read().decode("utf-8", errors="strict")


def http_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    text = http_text(url, timeout=timeout)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def http_status(url: str, *, timeout: int = 20) -> int:
    request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context()
    try:
        with urlopen(request, timeout=timeout, context=context) as response:
            return int(response.status)
    except HTTPError as exc:
        if exc.code == 405:
            get_req = Request(url, method="GET", headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
            with urlopen(get_req, timeout=timeout, context=context) as response:
                return int(response.status)
        return int(exc.code)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first(values: Iterable[str]) -> str | None:
    for value in values:
        if str(value).strip():
            return str(value)
    return None


def series_root(accession: str) -> str:
    n = int(accession[3:])
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{n // 1000}nnn/{accession}/"


def parse_geo_soft(text: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        if line.startswith("^SAMPLE = "):
            gsm = line.split("=", 1)[1].strip().upper()
            current = {"geo_accession": gsm}
            records[gsm] = current
            continue
        if current is None or not line.startswith("!Sample_") or " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        key = key[len("!Sample_"):]
        value = value.strip().replace('\\"', '"')
        if re.fullmatch(r"supplementary_file_\d+", key):
            key = "supplementary_file"
        if key in {"supplementary_file", "relation", "characteristics_ch1"}:
            current.setdefault(key, []).append(value)
        elif key not in current:
            current[key] = value
    return records


def geo_metadata(accession: str) -> tuple[dict[str, dict[str, Any]], str]:
    url = f"{series_root(accession)}soft/{accession}_family.soft.gz"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context()
    with urlopen(request, timeout=60, context=context) as response:
        text = gzip.decompress(response.read()).decode("utf-8", errors="strict")
    return parse_geo_soft(text), url


def geo_candidate_files(
    sample_records: list[dict[str, Any]],
    accession: str,
) -> tuple[list[str], list[str]]:
    candidates: set[str] = set()
    rejected: set[str] = set()
    base = f"{series_root(accession)}suppl/"
    for record in sample_records:
        values = record.get("supplementary_file", [])
        if not isinstance(values, list):
            continue
        for value in values:
            raw = str(value).strip()
            if not raw or raw.upper() in {"NONE", "NA", "NULL"}:
                continue
            url = raw
            if url.startswith("ftp://"):
                url = "https://" + url[6:]
            elif not url.startswith(("http://", "https://")):
                url = urljoin(base, raw.lstrip("/"))
            name = Path(urlparse(url).path).name.lower()
            if any(token in name for token in REJECT_TOKENS):
                rejected.add(url)
            elif any(token in name for token in RAW_TOKENS):
                candidates.add(url)
    return sorted(candidates), sorted(rejected)


def extract_sra_links(record: dict[str, Any]) -> list[str]:
    relations = record.get("relation", [])
    if not isinstance(relations, list):
        return []
    return [str(value) for value in relations if "sra" in str(value).lower()]


def extract_accessions(text: str, pattern: re.Pattern[str]) -> list[str]:
    return sorted({x.upper() for x in pattern.findall(text)})


def ena_run_report(srx: str) -> dict[str, Any]:
    params = {
        "accession": srx,
        "result": "read_run",
        "fields": "run_accession,read_count,base_count,fastq_ftp,fastq_bytes",
        "format": "json",
    }
    return http_json(f"{ENA_API}?{urlencode(params)}")


def atlas_search(gse: str) -> dict[str, Any]:
    params = {
        "query": gse,
        "organism": "Mus musculus",
        "pageSize": 100,
        "page": 1,
    }
    return http_json(f"{BIOSTUDIES_SEARCH}?{urlencode(params)}")


def study_file_candidates(study_json: dict[str, Any]) -> list[str]:
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in {"path", "file", "name", "url"} and isinstance(child, str):
                    if any(token in child.lower() for token in RAW_TOKENS):
                        found.append(child)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(study_json)
    return sorted(set(found))


def expression_atlas_evidence(gse: str, sample_ids: list[str]) -> SourceEvidence:
    try:
        search = atlas_search(gse)
    except Exception as exc:
        return SourceEvidence(
            source_type="expression_atlas_raw_counts",
            status="not_found",
            accession=gse,
            sample_ids=sample_ids,
            notes=[f"ArrayExpress/BioStudies search unavailable: {type(exc).__name__}: {exc}"],
        )

    hits = search.get("hits", [])
    if not isinstance(hits, list):
        hits = []

    experiment_hits = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        accession = str(hit.get("accession", "") or hit.get("id", "")).strip()
        if accession:
            experiment_hits.append(accession)

    candidate_files: set[str] = set()
    atlas_pages: list[str] = []
    matched_studies: list[str] = []

    for accession in sorted(set(experiment_hits)):
        try:
            study = http_json(f"{BIOSTUDIES_STUDY}/{quote(accession, safe='')}")
        except Exception:
            continue
        files = study_file_candidates(study)
        if files:
            matched_studies.append(accession)
            candidate_files.update(files)
        page = f"{ATLAS_EXPERIMENT}{quote(accession, safe='')}"
        try:
            if http_status(page) < 400:
                atlas_pages.append(page)
        except Exception:
            pass

    if candidate_files and atlas_pages:
        status = "candidate"
        notes = [
            "Expression Atlas page and raw/count-like ArrayExpress/BioStudies file evidence found.",
            "Exact GSM coverage still requires sample-level file reconciliation before approval.",
        ]
    elif atlas_pages:
        status = "candidate_metadata_only"
        notes = [
            "Expression Atlas experiment page found, but no raw/count-like file path was recovered from BioStudies metadata.",
            "Use the Atlas download contract to inspect its RNA-seq raw-count artifact.",
        ]
    else:
        status = "not_found"
        notes = ["No Expression Atlas experiment with direct evidence was resolved from the GSE query."]

    return SourceEvidence(
        source_type="expression_atlas_raw_counts",
        status=status,
        accession=gse,
        sample_ids=sample_ids,
        source_urls=atlas_pages,
        candidate_files=sorted(candidate_files),
        exact_sample_coverage=0,
        exact_sample_expected=len(sample_ids),
        notes=notes,
        provenance={
            "arrayexpress_hits": sorted(set(experiment_hits)),
            "matched_biostudies_studies": matched_studies,
        },
    )


def archs4_evidence(
    gse: str,
    sample_ids: list[str],
    h5_path: str | None,
) -> SourceEvidence:
    if not h5_path:
        return SourceEvidence(
            source_type="archs4_kallisto_rounded",
            status="not_checked",
            accession=gse,
            sample_ids=sample_ids,
            notes=[
                "No local ARCHS4 H5 supplied. Current mouse gene H5 is tens of GB; do not auto-download in the audit.",
                "When supplied, only metadata/sample availability is checked here.",
            ],
        )

    path = Path(h5_path)
    if not path.is_file():
        return SourceEvidence(
            source_type="archs4_kallisto_rounded",
            status="error",
            accession=gse,
            sample_ids=sample_ids,
            notes=[f"ARCHS4 H5 not found: {path}"],
        )

    try:
        with h5py.File(path, "r") as h5:
            values = h5["meta/samples/geo_accession"][()]
            available = {
                x.decode() if isinstance(x, bytes) else str(x)
                for x in values
            }
    except Exception as exc:
        return SourceEvidence(
            source_type="archs4_kallisto_rounded",
            status="error",
            accession=gse,
            sample_ids=sample_ids,
            notes=[f"Could not read ARCHS4 H5 metadata: {type(exc).__name__}: {exc}"],
        )

    matched = sorted(set(sample_ids) & available)
    return SourceEvidence(
        source_type="archs4_kallisto_rounded",
        status="available" if matched else "not_found",
        accession=gse,
        sample_ids=sample_ids,
        exact_sample_coverage=len(matched),
        exact_sample_expected=len(sample_ids),
        notes=[
            "ARCHS4 gene-level H5 values are Kallisto-derived pseudocounts rounded to integers; "
            "they are not promoted to strict raw-count status."
        ],
        provenance={
            "h5_sha256": sha256_file(path),
            "h5_path": str(path),
            "matched_gsms": matched,
        },
    )


def recount3_project_candidates(sra_links: Iterable[str]) -> list[str]:
    text = " ".join(map(str, sra_links))
    return sorted(set(PROJECT_RE.findall(text)))


def recount3_evidence(
    gse: str,
    sample_ids: list[str],
    sra_links: list[str],
) -> SourceEvidence:
    projects = recount3_project_candidates(sra_links)
    if not projects:
        return SourceEvidence(
            source_type="recount3_derived_read_counts",
            status="not_found",
            accession=gse,
            sample_ids=sample_ids,
            notes=["No SRP/ERP/DRP project identifier was present in the GEO relation fields."],
        )

    # First recover SRX -> SRR through ENA. This keeps the exact run mapping tied to the GEO GSM.
    srx_ids: set[str] = set()
    for relation in sra_links:
        srx_ids.update(extract_accessions(relation, SRX_RE))

    run_ids: set[str] = set()
    run_sizes: dict[str, Any] = {}
    for srx in sorted(srx_ids):
        try:
            payload = ena_run_report(srx)
        except Exception:
            continue
        reports = payload.get("records", [])
        if not isinstance(reports, list):
            continue
        for record in reports:
            if not isinstance(record, dict):
                continue
            run = str(record.get("run_accession", "")).strip().upper()
            if not run:
                continue
            run_ids.add(run)
            run_sizes[run] = {
                "read_count": record.get("read_count"),
                "base_count": record.get("base_count"),
                "fastq_bytes": record.get("fastq_bytes"),
                "fastq_ftp": record.get("fastq_ftp"),
                "srx": srx,
            }

    checked_urls: list[str] = []
    matched_runs: set[str] = set()

    for project in projects:
        suffix = project[-2:]
        for mirror in RECOUNT3_MIRRORS:
            url = (
                f"{mirror}/mouse/data_sources/sra/metadata/{suffix}/{project}/"
                f"sra.sra.{project}.MD.gz"
            )
            try:
                status = http_status(url)
            except Exception:
                continue
            if status >= 400:
                continue
            checked_urls.append(url)
            try:
                req = Request(url, headers={"User-Agent": USER_AGENT})
                with urlopen(req, timeout=60) as response:
                    blob = response.read()
                raw = gzip.decompress(blob).decode("utf-8", errors="replace")
                for run in run_ids:
                    if run in raw:
                        matched_runs.add(run)
            except Exception:
                continue

    if matched_runs:
        status = "candidate"
        notes = [
            "recount3 contains the resolved SRA runs at the project-metadata level.",
            "recount3 gene raw_counts are base-pair coverage counts; converting them to read-count estimates "
            "is derived and must not be labeled original submitter raw counts.",
        ]
    else:
        status = "project_found_but_sample_match_unconfirmed" if projects else "not_found"
        notes = [
            "recount3 project metadata was not sufficient to confirm exact run coverage from this audit."
        ]

    return SourceEvidence(
        source_type="recount3_derived_read_counts",
        status=status,
        accession=gse,
        sample_ids=sample_ids,
        source_urls=checked_urls,
        exact_sample_coverage=len(matched_runs),
        exact_sample_expected=len(run_ids),
        run_ids=sorted(run_ids),
        notes=notes,
        provenance={
            "projects": projects,
            "ena_run_sizes": run_sizes,
        },
    )


def strict_candidate_from_geo(
    accession: str,
    sample_records: list[dict[str, Any]],
) -> SourceEvidence:
    candidates, rejected = geo_candidate_files(sample_records, accession)
    gsm_ids = [str(x["geo_accession"]).upper() for x in sample_records]
    return SourceEvidence(
        source_type="geo_submitter_raw_counts",
        status="candidate" if candidates else "not_found",
        accession=accession,
        sample_ids=gsm_ids,
        source_urls=candidates,
        candidate_files=candidates,
        exact_sample_coverage=0,
        exact_sample_expected=len(gsm_ids),
        notes=[
            "Candidate identified from GEO supplementary filenames/metadata only.",
            "The file still requires content-scale and exact GSM-column validation before strict approval."
        ] + ([f"Rejected normalized-looking files: {len(rejected)}"] if rejected else []),
        provenance={"rejected_candidate_files": rejected},
    )


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["eligible_tracks"]["strict_biological_replicate_benchmark"])


def rank_recommendation(evidence: list[SourceEvidence]) -> dict[str, Any]:
    strict = [e for e in evidence if e.status in {"candidate", "available"} and e.source_type in STRICT_SOURCE_TYPES]
    derived = [e for e in evidence if e.status in {"candidate", "available"} and e.source_type in DERIVED_SOURCE_TYPES]
    archs4 = [e for e in evidence if e.source_type == "archs4_kallisto_rounded" and e.status == "available"]
    if strict:
        return {
            "recommended_action": "validate_strict_candidate",
            "source_type": strict[0].source_type,
            "reason": "A potentially strict raw-count source was discovered; verify exact sample coverage and count scale before approval.",
        }
    if derived:
        return {
            "recommended_action": "retain_as_derived_fallback",
            "source_type": derived[0].source_type,
            "reason": "recount3-derived evidence exists, but it is not equivalent to original submitter raw read counts.",
        }
    if archs4:
        return {
            "recommended_action": "do_not_promote_archs4",
            "source_type": "archs4_kallisto_rounded",
            "reason": "ARCHS4 sample coverage exists, but its gene-level values are rounded Kallisto pseudocounts.",
        }
    return {
        "recommended_action": "reprocess_sra",
        "source_type": "sra",
        "reason": "No acceptable strict source was discovered; reprocess raw reads with a pinned reference/counting pipeline.",
    }


def run_audit(manifest_path: Path, output_path: Path, archs4_h5: str | None = None) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    reports: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []

    for entry in manifest:
        accession = str(entry["accession"]).upper()
        sample_ids = [str(x).upper() for x in entry["samples"]]
        try:
            records, soft_url = geo_metadata(accession)
            selected = [records[gsm] for gsm in sample_ids if gsm in records]
            if len(selected) != len(sample_ids):
                missing = sorted(set(sample_ids) - set(records))
                raise ValueError(f"GEO SOFT missing locked GSMs: {missing}")
        except Exception as exc:
            reports[accession] = {
                "accession": accession,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "sources": [],
                "recommendation": {
                    "recommended_action": "manual_reconcile_geo",
                    "reason": "Could not retrieve/parse the authoritative GEO sample metadata."
                },
            }
            continue

        sra_links = sorted({link for record in selected for link in extract_sra_links(record)})
        srx_ids = sorted(set().union(*(set(extract_accessions(link, SRX_RE)) for link in sra_links)))

        sources = [
            strict_candidate_from_geo(accession, selected),
            expression_atlas_evidence(accession, sample_ids),
            archs4_evidence(accession, sample_ids, archs4_h5),
            recount3_evidence(accession, sample_ids, sra_links),
        ]

        run_evidence: list[dict[str, Any]] = []
        run_ids: set[str] = set()
        for srx in srx_ids:
            try:
                payload = ena_run_report(srx)
                records_ena = payload.get("records", [])
                if isinstance(records_ena, list):
                    for record in records_ena:
                        if isinstance(record, dict) and record.get("run_accession"):
                            run = str(record["run_accession"]).upper()
                            run_ids.add(run)
                            run_evidence.append(record)
            except Exception as exc:
                run_evidence.append({
                    "srx": srx,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        recommendation = rank_recommendation(sources)
        reports[accession] = {
            "accession": accession,
            "status": "audited",
            "geo_soft_url": soft_url,
            "sample_ids": sample_ids,
            "sra_relations": sra_links,
            "srx_ids": srx_ids,
            "srr_ids": sorted(run_ids),
            "srr_run_evidence": run_evidence,
            "sources": [asdict(x) for x in sources],
            "recommendation": recommendation,
        }

        for source in sources:
            all_rows.append({
                "accession": accession,
                "source_type": source.source_type,
                "status": source.status,
                "sample_coverage": source.exact_sample_coverage,
                "sample_expected": source.exact_sample_expected,
                "candidate_file_count": len(source.candidate_files),
                "run_count": len(source.run_ids),
                "notes": " | ".join(source.notes),
            })

    payload = {
        "schema_version": "1.0",
        "tool": "scripts/source_rescue_audit.py",
        "manifest": str(manifest_path),
        "strict_source_types": sorted(STRICT_SOURCE_TYPES),
        "derived_source_types": sorted(DERIVED_SOURCE_TYPES),
        "rejected_source_types": sorted(REJECT_SOURCE_TYPES),
        "accessions": reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit candidate raw-count sources for the locked CardiLearn benchmark")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="reports/source_rescue_v1.json")
    parser.add_argument(
        "--archs4-h5",
        default=None,
        help="Optional local ARCHS4 mouse gene H5; only metadata/sample availability is checked.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_audit(Path(args.manifest), Path(args.output), args.archs4_h5)
    print(json.dumps({
        accession: value.get("recommendation", {})
        for accession, value in payload["accessions"].items()
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
