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
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

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


def http_json(url: str, *, timeout: int = 30) -> Any:
    text = http_text(url, timeout=timeout)
    return json.loads(text)


def http_gzip_text(url: str, *, timeout: int = 60) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context()
    with urlopen(request, timeout=timeout, context=context) as response:
        return gzip.decompress(response.read()).decode("utf-8", errors="strict")


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


def geo_supplementary_listing(accession: str) -> list[str]:
    base = f"{series_root(accession)}suppl/"
    html = http_text(base, timeout=45)
    out: set[str] = set()
    for href in re.findall(r'href=["\']([^"\']+)["\']', html, re.I):
        if not href or href.startswith(("#", "?", "../", "./", "mailto:", "javascript:")):
            continue
        url = urljoin(base, href.strip())
        parsed = urlparse(url)
        if parsed.netloc != urlparse(base).netloc or not parsed.path.startswith(urlparse(base).path):
            continue
        name = Path(parsed.path).name
        if not name:
            continue
        low = name.lower()
        if low == "filelist.txt":
            continue
        if low.endswith((".txt", ".tsv", ".csv", ".tab", ".gz", ".tar", ".tar.gz", ".tgz", ".zip")):
            out.add(url)
    return sorted(out)


def _candidate_priority(url: str) -> tuple[int, str]:
    name = Path(urlparse(url).path).name.lower()
    score = 0
    for token in RAW_TOKENS:
        if token in name:
            score -= 10
    if any(token in name for token in REJECT_TOKENS):
        score += 10000
    return score, name


def geo_candidate_files(
    sample_records: list[dict[str, Any]],
    accession: str,
) -> tuple[list[str], list[str]]:
    explicit: set[str] = set()
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
            explicit.add(url)

    listing: list[str] = []
    try:
        listing = geo_supplementary_listing(accession)
    except Exception:
        listing = []

    all_urls = sorted(explicit | set(listing))
    candidates: list[str] = []
    for url in all_urls:
        name = Path(urlparse(url).path).name.lower()
        if any(token in name for token in REJECT_TOKENS):
            rejected.add(url)
            continue
        # Keep annotation/sequence archives out of expression-source discovery.
        if re.search(r"\.(?:gtf|gff3?|bed|fa|fasta|fna|bam|sam|cram|vcf|bw|bigwig)(?:\.gz)?$", name):
            continue
        if name.endswith((".txt", ".tsv", ".csv", ".tab", ".gz", ".tar", ".tar.gz", ".tgz", ".zip")):
            candidates.append(url)

    return sorted(set(candidates), key=_candidate_priority)[:100], sorted(rejected)


def extract_sra_links(record: dict[str, Any]) -> list[str]:
    relations = record.get("relation", [])
    if not isinstance(relations, list):
        return []
    return [str(value) for value in relations if "sra" in str(value).lower()]


def extract_accessions(text: str, pattern: re.Pattern[str]) -> list[str]:
    return sorted({x.upper() for x in pattern.findall(text)})


def ena_run_report(srx: str) -> list[dict[str, str]]:
    params = {
        "accession": srx,
        "result": "read_run",
        "fields": "study_accession,experiment_accession,run_accession,read_count,base_count,fastq_ftp,fastq_bytes",
        "format": "tsv",
    }
    text = http_text(f"{ENA_API}?{urlencode(params)}")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    headers = lines[0].split("\t")
    return [
        dict(zip(headers, line.split("\t")))
        for line in lines[1:]
        if line.strip()
    ]


def atlas_search(gse: str) -> Any:
    params = {
        "query": gse,
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


def _search_hits(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("hits", "results", "studies", "experiments", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    elif isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def expression_atlas_evidence(gse: str, sample_ids: list[str]) -> SourceEvidence:
    experiment_hits: list[str] = []
    search_errors: list[str] = []
    search_urls: list[str] = []

    search_urls_to_try = [
        f"{BIOSTUDIES_SEARCH}?{urlencode({'query': gse, 'pageSize': 100, 'page': 1})}",
        f"https://www.ebi.ac.uk/biostudies/api/v1/search?{urlencode({'query': gse, 'pageSize': 100, 'page': 1})}",
    ]
    for search_url in search_urls_to_try:
        try:
            payload = http_json(search_url)
            hits = _search_hits(payload)
            for hit in hits:
                accession = str(
                    hit.get("accession")
                    or hit.get("id")
                    or hit.get("acc")
                    or hit.get("accessionCode")
                    or ""
                ).strip()
                if accession:
                    experiment_hits.append(accession)
            search_urls.append(search_url)
            if experiment_hits:
                break
        except Exception as exc:
            search_errors.append(f"{type(exc).__name__}: {exc}")

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

    # Expression Atlas is derived from BioStudies/ArrayExpress; its current download
    # contract states that RNA-seq experiments expose an "raw counts" download generated
    # by htseq-count. Finding the experiment page is evidence, but exact GSM coverage and
    # the downloaded count file still require validation.
    if atlas_pages or candidate_files:
        status = "candidate"
        notes = [
            "EBI experiment/BioStudies evidence found.",
            "Expression Atlas RNA-seq raw-count files are documented as htseq-count outputs; "
            "exact sample coverage and file content must still be validated before strict approval.",
        ]
    else:
        status = "not_found"
        notes = ["No EBI ArrayExpress/BioStudies/Atlas experiment was resolved from the GSE query."]
    notes.extend(search_errors)

    return SourceEvidence(
        source_type="expression_atlas_raw_counts",
        status=status,
        accession=gse,
        sample_ids=sample_ids,
        source_urls=search_urls + atlas_pages,
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
    if h5_path:
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
            import h5py
        except ImportError:
            return SourceEvidence(
                source_type="archs4_kallisto_rounded",
                status="not_checked_dependency_missing",
                accession=gse,
                sample_ids=sample_ids,
                notes=["Install h5py to inspect the supplied ARCHS4 H5 file."],
            )
        try:
            with h5py.File(path, "r") as h5:
                values = h5["meta/samples/geo_accession"][()]
                available = {
                    (x.decode() if isinstance(x, bytes) else str(x)).upper()
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
        status = "available" if len(matched) == len(set(sample_ids)) else ("partial" if matched else "not_found")
        return SourceEvidence(
            source_type="archs4_kallisto_rounded",
            status=status,
            accession=gse,
            sample_ids=sample_ids,
            exact_sample_coverage=len(matched),
            exact_sample_expected=len(sample_ids),
            notes=[
                "ARCHS4 gene-level values are Kallisto-derived/rounded and are never promoted to strict raw-count status."
            ],
            provenance={"h5_sha256": sha256_file(path), "h5_path": str(path), "matched_gsms": matched},
        )

    url = f"https://maayanlab.cloud/archs4/series/{gse}"
    try:
        html = http_text(url, timeout=45)
    except Exception as exc:
        return SourceEvidence(
            source_type="archs4_kallisto_rounded",
            status="not_found",
            accession=gse,
            sample_ids=sample_ids,
            source_urls=[url],
            notes=[f"ARCHS4 series page unavailable: {type(exc).__name__}: {exc}"],
        )

    available = {x.upper() for x in GSM_RE.findall(html)}
    matched = sorted(set(sample_ids) & available)
    status = "available" if len(matched) == len(set(sample_ids)) else ("partial" if matched else "not_found")
    return SourceEvidence(
        source_type="archs4_kallisto_rounded",
        status=status,
        accession=gse,
        sample_ids=sample_ids,
        source_urls=[url],
        exact_sample_coverage=len(matched),
        exact_sample_expected=len(sample_ids),
        notes=[
            "ARCHS4 series-page lookup performed without downloading the large H5 database.",
            "ARCHS4 gene-level values are Kallisto-derived/rounded and are never promoted to strict raw-count status.",
        ],
        provenance={"series_url": url, "matched_gsms": matched},
    )


def recount3_project_candidates(accessions: Iterable[str]) -> list[str]:
    projects = set()
    for accession in accessions:
        value = str(accession).strip().upper()
        if PROJECT_RE.fullmatch(value) or value.startswith("PRJNA"):
            projects.add(value)
    return sorted(projects)


def recount3_master_samples(organism: str = "mouse") -> list[dict[str, str]]:
    urls = [
        f"https://duffel.rail.bio/recount3/{organism}/data_sources/sra/metadata/sra.recount_project.MD.gz",
        f"http://duffel.rail.bio/recount3/{organism}/data_sources/sra/metadata/sra.recount_project.MD.gz",
        f"https://recount-opendata.s3.amazonaws.com/recount3/release/{organism}/data_sources/sra/metadata/sra.recount_project.MD.gz",
    ]
    last_error: Exception | None = None
    for url in urls:
        try:
            text = http_gzip_text(url, timeout=90)
            lines = [line for line in text.splitlines() if line.strip()]
            if not lines:
                return []
            headers = [h.strip() for h in lines[0].split("\t")]
            rows: list[dict[str, str]] = []
            for line in lines[1:]:
                values = line.split("\t")
                if len(values) != len(headers):
                    continue
                rows.append(dict(zip(headers, values)))
            return rows
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return []


def recount3_evidence(
    gse: str,
    sample_ids: list[str],
    sra_links: list[str],
    ena_records: list[dict[str, str]] | None = None,
) -> SourceEvidence:
    records = ena_records or []
    run_ids = sorted({
        str(record.get("run_accession", "")).strip().upper()
        for record in records
        if record.get("run_accession")
    })
    if not run_ids:
        return SourceEvidence(
            source_type="recount3_derived_read_counts",
            status="not_found",
            accession=gse,
            sample_ids=sample_ids,
            notes=["No SRR run accessions were resolved from ENA for the locked SRX records."],
        )

    try:
        master = recount3_master_samples("mouse")
    except Exception as exc:
        return SourceEvidence(
            source_type="recount3_derived_read_counts",
            status="unavailable",
            accession=gse,
            sample_ids=sample_ids,
            run_ids=run_ids,
            exact_sample_expected=len(run_ids),
            notes=[f"recount3 master sample index unavailable: {type(exc).__name__}: {exc}"],
        )

    external_key = next(
        (key for key in master[0].keys() if key == "external_id" or key.endswith(".external_id")),
        None,
    ) if master else None
    project_key = next(
        (key for key in master[0].keys() if key == "project" or key.endswith(".project")),
        None,
    ) if master else None
    if external_key is None:
        return SourceEvidence(
            source_type="recount3_derived_read_counts",
            status="unavailable",
            accession=gse,
            sample_ids=sample_ids,
            run_ids=run_ids,
            exact_sample_expected=len(run_ids),
            notes=["recount3 master sample index did not expose an external_id column."],
        )

    indexed = {
        str(row[external_key]).strip().upper(): row
        for row in master
        if row.get(external_key)
    }
    matched_runs = sorted(set(run_ids) & set(indexed))
    projects = sorted({
        str(indexed[run].get(project_key, "")).strip().upper()
        for run in matched_runs
        if project_key and indexed[run].get(project_key)
    })
    missing_runs = sorted(set(run_ids) - set(matched_runs))

    if len(matched_runs) == len(run_ids):
        status = "candidate"
    elif matched_runs:
        status = "partial"
    else:
        status = "not_found"

    notes = [
        "recount3 master sample index was queried directly by exact SRR external_id.",
        "recount3 gene raw_counts are base-pair coverage counts; any read-count conversion is derived and must not be labeled original submitter raw counts.",
    ]
    if missing_runs:
        notes.append(f"{len(missing_runs)} resolved SRR runs were absent from the current recount3 mouse index.")

    return SourceEvidence(
        source_type="recount3_derived_read_counts",
        status=status,
        accession=gse,
        sample_ids=sample_ids,
        source_urls=[
            "https://duffel.rail.bio/recount3/mouse/data_sources/sra/metadata/sra.recount_project.MD.gz"
        ],
        exact_sample_coverage=len(matched_runs),
        exact_sample_expected=len(run_ids),
        run_ids=run_ids,
        notes=notes,
        provenance={
            "projects": projects,
            "matched_runs": matched_runs,
            "missing_runs": missing_runs,
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
    strict = [
        e for e in evidence
        if e.status in {"candidate", "available"}
        and e.source_type in STRICT_SOURCE_TYPES
    ]
    derived = [
        e for e in evidence
        if e.status in {"candidate", "available"}
        and e.source_type in DERIVED_SOURCE_TYPES
    ]
    archs4 = [
        e for e in evidence
        if e.source_type == "archs4_kallisto_rounded"
        and e.status in {"available", "partial"}
    ]
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
            "reason": "recount3 contains the exact SRA runs, but its counts are derived from uniformly processed coverage rather than the original submitter count table.",
        }
    if archs4:
        return {
            "recommended_action": "do_not_promote_archs4",
            "source_type": "archs4_kallisto_rounded",
            "reason": "ARCHS4 sample coverage exists, but its gene-level values are Kallisto-derived/rounded and do not satisfy the strict raw-count contract.",
        }
    return {
        "recommended_action": "reprocess_sra",
        "source_type": "sra",
        "reason": "No acceptable strict or derived source was discovered; reprocess raw reads with a pinned reference/counting pipeline.",
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
                    "reason": "Could not retrieve/parse the authoritative GEO sample metadata.",
                },
            }
            continue

        sra_links = sorted({link for record in selected for link in extract_sra_links(record)})
        srx_ids = sorted({
            accession_id
            for link in sra_links
            for accession_id in extract_accessions(link, SRX_RE)
        })

        run_evidence: list[dict[str, str]] = []
        for srx in srx_ids:
            try:
                run_evidence.extend(ena_run_report(srx))
            except Exception as exc:
                run_evidence.append({
                    "experiment_accession": srx,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        sources = [
            strict_candidate_from_geo(accession, selected),
            expression_atlas_evidence(accession, sample_ids),
            archs4_evidence(accession, sample_ids, archs4_h5),
            recount3_evidence(accession, sample_ids, sra_links, run_evidence),
        ]
        run_ids = sorted({
            str(record.get("run_accession")).strip().upper()
            for record in run_evidence
            if record.get("run_accession")
        })

        recommendation = rank_recommendation(sources)
        reports[accession] = {
            "accession": accession,
            "status": "audited",
            "geo_soft_url": soft_url,
            "sample_ids": sample_ids,
            "sra_relations": sra_links,
            "srx_ids": srx_ids,
            "srr_ids": run_ids,
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
        "schema_version": "1.1",
        "tool": "scripts/source_rescue_audit.py",
        "manifest": str(manifest_path),
        "strict_source_types": sorted(STRICT_SOURCE_TYPES),
        "derived_source_types": sorted(DERIVED_SOURCE_TYPES),
        "rejected_source_types": sorted(REJECT_SOURCE_TYPES),
        "accessions": reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(output_path.with_suffix(".csv"), index=False)
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
