#!/usr/bin/env python3
"""Reproducible raw-read rescue for the locked CardiLearn benchmark.

The default source is ENA-generated FASTQ for the exact SRRs linked to each locked
GEO GSM. SRA Toolkit is retained as a fallback for runs without ENA FASTQ URLs.

The pipeline is intentionally fail-closed:
    resolve -> download -> concat -> qc -> reference -> align -> count -> merge -> validate

It never rounds transformed expression values. Only featureCounts integer gene counts
generated from the raw reads can be emitted as a Step-3 external source.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

GEO_SOFT = "https://ftp.ncbi.nlm.nih.gov/geo/series"
ENA_FILEREPORT = "https://www.ebi.ac.uk/ena/portal/api/filereport"
USER_AGENT = "Virelion-CardiLearn/SRA-reprocess-v1"

ACCESSION_RE = re.compile(r"^GSE\d+$", re.I)
GSM_RE = re.compile(r"^GSM\d+$", re.I)
SRX_RE = re.compile(r"\b(?:SRX|ERX|DRX)\d+\b", re.I)
FASTQ_EXT_RE = re.compile(r"(?:\.fastq|\.fq)(?:\.gz)?$", re.I)
INTEGER_RE = re.compile(r"^[+]?\d+$")


def die(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_bytes(url: str, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def http_text(url: str, timeout: int = 120) -> str:
    return http_bytes(url, timeout).decode("utf-8", errors="strict")


def download_resume(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        # curl handles HTTP range/resume and FTP/HTTPS redirects more reliably than
        # reimplementing a resumable downloader in Python.
        cmd = ["curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
               "--continue-at", "-", "--output", str(dest), url]
    else:
        cmd = ["curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
               "--output", str(dest), url]
    run(cmd)


def run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode:
        die(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def write_done(path: Path, payload: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload or {}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def is_done(path: Path) -> bool:
    return path.is_file()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        die(f"could not parse JSON {path}: {type(exc).__name__}: {exc}")
    if not isinstance(payload, dict):
        die(f"JSON root must be an object: {path}")
    return payload


def config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "sra_reprocessing_v1.json"


def load_config(repo_root: Path) -> dict[str, Any]:
    config = load_json(config_path(repo_root))
    if config.get("schema_version") != "1.0":
        die("unsupported SRA rescue config schema")
    return config


def manifest_entries(repo_root: Path, selected: set[str]) -> list[dict[str, Any]]:
    path = repo_root / "data" / "manifest.lock.json"
    manifest = load_json(path)
    entries = manifest["eligible_tracks"]["strict_biological_replicate_benchmark"]
    selected_entries = [
        entry for entry in entries
        if str(entry["accession"]).upper() in selected
    ]
    missing = sorted(selected - {str(e["accession"]).upper() for e in selected_entries})
    if missing:
        die(f"requested accessions are absent from manifest: {missing}")
    return selected_entries


def series_root(accession: str) -> str:
    n = int(accession[3:])
    return f"{GEO_SOFT}/GSE{n // 1000}nnn/{accession}/"


def parse_soft(text: str) -> dict[str, dict[str, Any]]:
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
        if key in {"relation", "characteristics_ch1", "supplementary_file"}:
            current.setdefault(key, []).append(value)
        elif key not in current:
            current[key] = value
    return records


def fetch_geo_records(accession: str) -> dict[str, dict[str, Any]]:
    url = f"{series_root(accession)}soft/{accession}_family.soft.gz"
    raw = http_bytes(url, timeout=120)
    return parse_soft(gzip.decompress(raw).decode("utf-8", errors="strict"))


def sra_relations(record: dict[str, Any]) -> list[str]:
    return [
        str(x) for x in record.get("relation", [])
        if "sra" in str(x).lower()
    ]


def extract_srx(relations: Iterable[str]) -> list[str]:
    return sorted({
        match.upper()
        for relation in relations
        for match in SRX_RE.findall(str(relation))
    })


def ena_rows(srx: str) -> list[dict[str, str]]:
    fields = ",".join([
        "study_accession",
        "experiment_accession",
        "run_accession",
        "sample_accession",
        "library_layout",
        "library_strategy",
        "instrument_platform",
        "instrument_model",
        "read_count",
        "base_count",
        "fastq_ftp",
        "fastq_md5",
        "fastq_bytes",
    ])
    params = {
        "accession": srx,
        "result": "read_run",
        "fields": fields,
        "format": "tsv",
    }
    body = http_text(f"{ENA_FILEREPORT}?{urlencode(params)}", timeout=120)
    lines = [line for line in body.splitlines() if line.strip()]
    if not lines:
        return []
    headers = lines[0].split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        values = line.split("\t")
        if len(values) != len(headers):
            die(f"ENA response row has {len(values)} values but {len(headers)} headers for {srx}")
        rows.append(dict(zip(headers, values)))
    return rows


def resolve(repo_root: Path, out_root: Path, accessions: set[str]) -> Path:
    entries = manifest_entries(repo_root, accessions)
    out_root.mkdir(parents=True, exist_ok=True)
    meta = out_root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for entry in entries:
        accession = str(entry["accession"]).upper()
        locked_gsms = [str(x).upper() for x in entry["samples"]]
        geo = fetch_geo_records(accession)
        for gsm in locked_gsms:
            if gsm not in geo:
                die(f"{accession}: locked GSM missing from GEO SOFT: {gsm}")
            relations = sra_relations(geo[gsm])
            srx_ids = extract_srx(relations)
            if len(srx_ids) != 1:
                die(f"{accession}/{gsm}: expected exactly one SRA experiment relation, found {srx_ids}")
            srx = srx_ids[0]
            ena = ena_rows(srx)
            if not ena:
                die(f"{accession}/{gsm}/{srx}: ENA returned no run rows")
            layouts = {str(r.get("library_layout", "")).upper() for r in ena}
            layouts.discard("")
            if len(layouts) > 1:
                die(f"{accession}/{gsm}/{srx}: mixed library layouts across runs: {sorted(layouts)}")
            for run_index, record in enumerate(ena, start=1):
                run_accession = str(record.get("run_accession", "")).strip().upper()
                if not run_accession:
                    die(f"{accession}/{gsm}/{srx}: ENA row lacks run_accession")
                urls = [x for x in str(record.get("fastq_ftp", "")).split(";") if x]
                md5s = [x for x in str(record.get("fastq_md5", "")).split(";") if x]
                sizes = [x for x in str(record.get("fastq_bytes", "")).split(";") if x]
                if not urls:
                    # Keep the row so the download stage can fall back to SRA Toolkit.
                    urls = [""]
                if md5s and len(md5s) not in {1, len(urls)}:
                    die(f"{run_accession}: fastq_md5 count does not match fastq_ftp")
                if sizes and len(sizes) not in {1, len(urls)}:
                    die(f"{run_accession}: fastq_bytes count does not match fastq_ftp")
                for file_index, url in enumerate(urls, start=1):
                    rows.append({
                        "accession": accession,
                        "gsm": gsm,
                        "srx": srx,
                        "srr": run_accession,
                        "run_index": str(run_index),
                        "file_index": str(file_index),
                        "library_layout": next(iter(layouts)) if layouts else "",
                        "library_strategy": str(record.get("library_strategy", "")),
                        "instrument_platform": str(record.get("instrument_platform", "")),
                        "instrument_model": str(record.get("instrument_model", "")),
                        "read_count": str(record.get("read_count", "")),
                        "base_count": str(record.get("base_count", "")),
                        "fastq_ftp": url,
                        "fastq_md5": (
                            md5s[file_index - 1] if len(md5s) == len(urls)
                            else (md5s[0] if md5s else "")
                        ),
                        "fastq_bytes": (
                            sizes[file_index - 1] if len(sizes) == len(urls)
                            else (sizes[0] if sizes else "")
                        ),
                    })

    # One row per FASTQ file, with exact GEO->SRA->ENA lineage.
    fields = list(rows[0].keys())
    tsv = meta / "locked_run_manifest.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": "1.0",
        "manifest_sha256": sha256(repo_root / "data" / "manifest.lock.json"),
        "config_sha256": sha256(config_path(repo_root)),
        "accessions": sorted(accessions),
        "fastq_file_rows": len(rows),
        "rows": rows,
    }
    json_path = meta / "locked_run_manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (meta / "locked_run_manifest.sha256").write_text(sha256(json_path) + "\n", encoding="utf-8")
    return tsv


def load_run_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fastq_dest(run: dict[str, str], raw_root: Path) -> Path:
    url = run["fastq_ftp"]
    name = Path(url.split("?", 1)[0]).name if url else f'{run["srr"]}.fastq.gz'
    if not name:
        name = f'{run["srr"]}.{run["file_index"]}.fastq.gz'
    return raw_root / run["accession"] / run["gsm"] / run["srr"] / name


def normalize_ftp(url: str) -> str:
    if url.startswith("ftp://"):
        return "https://" + url[6:]
    return url


def verify_expected_md5(path: Path, expected: str, fail_on_missing: bool) -> str:
    if not expected:
        if fail_on_missing:
            die(f"missing expected ENA MD5 for {path}")
        return md5(path)
    actual = md5(path)
    if actual.lower() != expected.strip().lower():
        die(f"MD5 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def download_fastq(run_manifest: Path, out_root: Path, config: dict[str, Any]) -> None:
    rows = load_run_manifest(run_manifest)
    raw_root = out_root / "raw"
    fail_on_missing = bool(config["retrieval"]["fail_on_missing_md5"])
    for row in rows:
        dest = fastq_dest(row, raw_root)
        if row["fastq_ftp"]:
            download_resume(normalize_ftp(row["fastq_ftp"]), dest)
            actual_md5 = verify_expected_md5(dest, row["fastq_md5"], fail_on_missing)
            (dest.with_suffix(dest.suffix + ".md5")).write_text(actual_md5 + "\n", encoding="utf-8")
        else:
            # Fallback: NCBI SRA Toolkit.
            srr_dir = out_root / "sra" / row["srr"]
            srr_dir.mkdir(parents=True, exist_ok=True)
            run(["prefetch", row["srr"], "--max-size", "u", "-O", str(srr_dir.parent)])
            run([
                "fasterq-dump",
                row["srr"],
                "--split-files",
                "--threads", str(config["resources"]["threads_default"]),
                "--temp", str(out_root / "tmp"),
                "--outdir", str(srr_dir),
            ])
            for fq in sorted(srr_dir.glob(f"{row['srr']}*.fastq")):
                gz = Path(str(fq) + ".gz")
                with gzip.open(gz, "wb") as out, fq.open("rb") as src:
                    shutil.copyfileobj(src, out)
                fq.unlink()


def raw_rows(run_manifest: Path, accession: str, gsm: str) -> list[dict[str, str]]:
    return [
        row for row in load_run_manifest(run_manifest)
        if row["accession"] == accession and row["gsm"] == gsm
    ]


def concatenate_sample(run_manifest: Path, out_root: Path, accession: str, gsm: str) -> tuple[Path, Path | None, str]:
    rows = raw_rows(run_manifest, accession, gsm)
    if not rows:
        die(f"no run rows for {accession}/{gsm}")
    layout = rows[0]["library_layout"].upper()
    if any(row["library_layout"].upper() != layout for row in rows):
        die(f"{accession}/{gsm}: inconsistent library layout")
    if layout not in {"PAIRED", "SINGLE"}:
        die(f"{accession}/{gsm}: unsupported/unknown library layout {layout!r}")

    raw_root = out_root / "raw"
    merged = out_root / "samples" / accession / gsm
    merged.mkdir(parents=True, exist_ok=True)
    files_by_end: dict[str, list[Path]] = {"single": [], "r1": [], "r2": []}
    for row in rows:
        dest = fastq_dest(row, raw_root)
        if not dest.exists():
            # Fallback outputs are named by SRR and mate.
            if row["library_layout"].upper() == "PAIRED":
                candidates = sorted((out_root / "sra" / row["srr"]).glob(f"{row['srr']}_*.fastq.gz"))
            else:
                candidates = sorted((out_root / "sra" / row["srr"]).glob(f"{row['srr']}.fastq.gz"))
            if not candidates:
                die(f"downloaded FASTQ missing for {row['srr']}")
            # For fallback rows, use file_index to select the requested file.
            idx = int(row["file_index"]) - 1
            if idx >= len(candidates):
                die(f"fallback FASTQ index out of range for {row['srr']}")
            dest = candidates[idx]
        name = dest.name.lower()
        if layout == "SINGLE":
            files_by_end["single"].append(dest)
        else:
            if re.search(r"(?:_1|\.1)(?:\.fastq|\.fq)(?:\.gz)?$", name):
                files_by_end["r1"].append(dest)
            elif re.search(r"(?:_2|\.2)(?:\.fastq|\.fq)(?:\.gz)?$", name):
                files_by_end["r2"].append(dest)
            else:
                die(f"cannot determine mate for paired FASTQ: {dest}")

    if layout == "PAIRED" and len(files_by_end["r1"]) != len(files_by_end["r2"]):
        die(f"{accession}/{gsm}: paired FASTQ mate count mismatch")

    single_out = merged / f"{gsm}.fastq.gz"
    r1_out = merged / f"{gsm}_R1.fastq.gz"
    r2_out = merged / f"{gsm}_R2.fastq.gz"

    if layout == "SINGLE":
        run(["bash", "-lc", "cat " + " ".join("'" + str(p) + "'" for p in files_by_end["single"]) +
             " > '" + str(single_out) + "'"])
        return single_out, None, layout

    run(["bash", "-lc", "cat " + " ".join("'" + str(p) + "'" for p in files_by_end["r1"]) +
         " > '" + str(r1_out) + "'"])
    run(["bash", "-lc", "cat " + " ".join("'" + str(p) + "'" for p in files_by_end["r2"]) +
         " > '" + str(r2_out) + "'"])
    return r1_out, r2_out, layout


def run_fastp(
    r1: Path,
    r2: Path | None,
    out_root: Path,
    accession: str,
    gsm: str,
    config: dict[str, Any],
) -> tuple[Path, Path | None]:
    qc = out_root / "qc" / accession
    qc.mkdir(parents=True, exist_ok=True)
    html = qc / f"{gsm}.fastp.html"
    report_json = qc / f"{gsm}.fastp.json"
    done = qc / f"{gsm}.done.json"
    paired = r2 is not None
    out1 = qc / f"{gsm}_R1.fastq.gz" if paired else qc / f"{gsm}.fastq.gz"
    out2 = qc / f"{gsm}_R2.fastq.gz" if paired else None

    if is_done(done) and out1.exists() and (out2 is None or out2.exists()) and report_json.exists():
        return out1, out2

    args = [
        "fastp",
        "-w", str(config["resources"]["threads_default"]),
        "-j", str(report_json),
        "-h", str(html),
    ]
    if paired:
        args += ["-i", str(r1), "-I", str(r2), "-o", str(out1), "-O", str(out2)]
    else:
        args += ["-i", str(r1), "-o", str(out1)]

    qc_cfg = config["qc"]
    if not qc_cfg["trim_and_filter"]:
        args += [
            "--disable_adapter_trimming",
            "--disable_quality_filtering",
            "--disable_length_filtering",
            "--disable_trim_poly_g",
            "--disable_trim_poly_x",
        ]
    else:
        args += [
            "--cut_front",
            "--cut_tail",
            "--cut_mean_quality", str(qc_cfg["quality_cutoff"]),
            "--length_required", str(qc_cfg["minimum_read_length"]),
        ]
    run(args)
    write_done(done, {
        "input_r1_sha256": sha256(r1),
        "input_r2_sha256": sha256(r2) if r2 else None,
    })
    return out1, out2


def reference(repo_root: Path, out_root: Path, config: dict[str, Any], threads: int) -> tuple[Path, Path, Path]:
    ref = out_root / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    fasta_gz = ref / Path(config["reference"]["fasta_gz"]).name
    gtf_gz = ref / Path(config["reference"]["gtf_gz"]).name
    if not fasta_gz.exists():
        download_resume(config["reference"]["fasta_gz"], fasta_gz)
    if not gtf_gz.exists():
        download_resume(config["reference"]["gtf_gz"], gtf_gz)

    fasta_sha = sha256(fasta_gz)
    gtf_sha = sha256(gtf_gz)
    configured_fasta = config["reference"].get("fasta_sha256")
    configured_gtf = config["reference"].get("gtf_sha256")
    if configured_fasta and configured_fasta.lower() != fasta_sha.lower():
        die(f"reference FASTA SHA256 mismatch: expected {configured_fasta}, got {fasta_sha}")
    if configured_gtf and configured_gtf.lower() != gtf_sha.lower():
        die(f"reference GTF SHA256 mismatch: expected {configured_gtf}, got {gtf_sha}")

    fasta = ref / "Mus_musculus.GRCm39.dna.primary_assembly.fa"
    gtf = ref / "Mus_musculus.GRCm39.112.gtf"
    if not fasta.exists():
        run(["bash", "-lc", f"gzip -dc '{fasta_gz}' > '{fasta}'"])
    if not gtf.exists():
        run(["bash", "-lc", f"gzip -dc '{gtf_gz}' > '{gtf}'"])

    index = ref / "STAR_index"
    if not (index / "Genome").exists():
        index.mkdir(parents=True, exist_ok=True)
        run([
            "STAR",
            "--runMode", "genomeGenerate",
            "--runThreadN", str(threads),
            "--genomeDir", str(index),
            "--genomeFastaFiles", str(fasta),
            "--sjdbGTFfile", str(gtf),
            "--sjdbOverhang", str(config["reference"]["sjdb_overhang"]),
        ])

    fingerprint = {
        "assembly": config["reference"]["assembly"],
        "species": config["reference"]["species"],
        "ensembl_release": config["reference"]["ensembl_release"],
        "fasta_gz_sha256": fasta_sha,
        "gtf_gz_sha256": gtf_sha,
        "fasta_sha256": sha256(fasta),
        "gtf_sha256": sha256(gtf),
        "star_index_dir": str(index),
        "sjdb_overhang": config["reference"]["sjdb_overhang"],
    }
    (ref / "reference_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index, fasta, gtf


def align(
    input_r1: Path,
    input_r2: Path | None,
    out_root: Path,
    accession: str,
    gsm: str,
    star_index: Path,
    threads: int,
) -> Path:
    outdir = out_root / "alignments" / accession / gsm
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / f"{gsm}."
    bam = outdir / f"{gsm}.Aligned.sortedByCoord.out.bam"
    done = outdir / "alignment.done.json"
    if is_done(done) and bam.exists():
        return bam

    cmd = [
        "STAR",
        "--runThreadN", str(threads),
        "--genomeDir", str(star_index),
        "--readFilesCommand", "zcat",
        "--readFilesIn", str(input_r1),
    ]
    if input_r2 is not None:
        cmd.append(str(input_r2))
    cmd += [
        "--twopassMode", "Basic",
        "--genomeLoad", "NoSharedMemory",
        "--outSAMtype", "BAM", "SortedByCoordinate",
        "--outFileNamePrefix", str(prefix),
        "--outSAMstrandField", "intronMotif",
        "--outFilterMultimapNmax", "20",
        "--outSAMunmapped", "Within",
        "--limitBAMsortRAM", str(30 * 1024 * 1024 * 1024),
    ]
    run(cmd)
    if not bam.exists():
        die(f"STAR completed without expected BAM: {bam}")
    write_done(done, {"bam_sha256": sha256(bam)})
    return bam


def feature_count(
    bam: Path,
    gtf: Path,
    out_root: Path,
    accession: str,
    gsm: str,
    paired: bool,
    threads: int,
    config: dict[str, Any],
) -> Path:
    outdir = out_root / "counts" / accession
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{gsm}.featureCounts.txt"
    done = outdir / f"{gsm}.done.json"
    if is_done(done) and output.exists():
        return output

    cmd = [
        "featureCounts",
        "-T", str(threads),
        "-a", str(gtf),
        "-o", str(output),
        "-t", str(config["counting"]["feature_type"]),
        "-g", str(config["counting"]["gene_id_attribute"]),
        "-s", str(config["counting"]["strand"]),
    ]
    if paired:
        cmd += ["-p", "--countReadPairs"]
    run(cmd + [str(bam)])
    if not output.exists():
        die(f"featureCounts completed without expected output: {output}")
    write_done(done, {
        "counts_sha256": sha256(output),
        "paired": paired,
    })
    return output


def parse_featurecounts(path: Path) -> tuple[str, dict[str, int]]:
    sample_col: int | None = None
    rows: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields and fields[0] == "Geneid":
                if len(fields) < 7:
                    die(f"featureCounts header too short: {path}")
                sample_col = len(fields) - 1
                continue
            if sample_col is None:
                continue
            if len(fields) <= sample_col:
                die(f"malformed featureCounts row in {path}")
            gene = fields[0].strip()
            value = fields[sample_col].strip()
            if not gene:
                continue
            if not INTEGER_RE.fullmatch(value):
                die(f"non-integer featureCounts value {value!r} for {gene} in {path}")
            if gene in rows:
                die(f"duplicate gene identifier {gene} in {path}")
            rows[gene] = int(value)
    if sample_col is None or not rows:
        die(f"no gene count rows parsed from {path}")
    return path.stem.split(".featureCounts", 1)[0], rows


def merge_counts(out_root: Path, accession: str, sample_ids: list[str]) -> Path:
    files = [out_root / "counts" / accession / f"{gsm}.featureCounts.txt" for gsm in sample_ids]
    for path in files:
        if not path.exists():
            die(f"missing featureCounts file for locked GSM: {path}")
    parsed = [parse_featurecounts(path) for path in files]
    gene_sets = [set(data) for _, data in parsed]
    genes = sorted(set.intersection(*gene_sets))
    if not genes:
        die(f"{accession}: no common genes across sample count files")
    outdir = out_root / "matrices"
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{accession}.raw_counts.tsv.gz"
    if output.exists():
        return output
    with gzip.open(output, "wt", encoding="utf-8", newline="") as handle:
        handle.write("gene_id\t" + "\t".join(sample_ids) + "\n")
        for gene in genes:
            handle.write(
                gene + "\t" +
                "\t".join(str(data[gene]) for _, data in parsed) +
                "\n"
            )
    return output


def validate_matrix(
    matrix: Path,
    repo_root: Path,
    accession: str,
    expected_samples: list[str],
    out_root: Path,
) -> None:
    with gzip.open(matrix, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if not header or header[0] != "gene_id":
            die(f"{accession}: invalid matrix header")
        observed = header[1:]
        if observed != expected_samples:
            die(f"{accession}: sample columns do not exactly match locked order")
        if len(observed) != len(set(observed)):
            die(f"{accession}: duplicate sample columns")
        n_genes = 0
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                die(f"{accession}: malformed matrix row {n_genes + 1}")
            gene = fields[0]
            if not re.fullmatch(r"ENSMUSG\d+", gene):
                die(f"{accession}: non-Ensembl gene identifier {gene!r}")
            for value in fields[1:]:
                if not INTEGER_RE.fullmatch(value):
                    die(f"{accession}: non-integer count {value!r} for {gene}")
                if int(value) < 0:
                    die(f"{accession}: negative count {value!r} for {gene}")
            n_genes += 1
        if n_genes == 0:
            die(f"{accession}: empty matrix")

    report_dir = out_root / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "accession": accession,
        "source_type": "sra_reprocessed_raw_counts",
        "source": str(matrix),
        "sha256": sha256(matrix),
        "manifest_sha256": sha256(repo_root / "data" / "manifest.lock.json"),
        "config_sha256": sha256(config_path(repo_root)),
        "n_samples": len(expected_samples),
        "samples": expected_samples,
        "n_genes": n_genes,
        "strict_contract": {
            "integer_like": True,
            "nonnegative": True,
            "duplicate_gene_ids": False,
            "duplicate_sample_ids": False,
            "all_locked_samples_present": True,
            "unexpected_samples": False,
        },
    }
    (report_dir / f"{accession}.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_source_registry(out_root: Path, accessions: list[str]) -> Path:
    sources: dict[str, dict[str, str]] = {}
    for accession in accessions:
        report = load_json(out_root / "validation" / f"{accession}.json")
        sources[accession] = {
            "source": report["source"],
            "sha256": report["sha256"],
            "source_type": report["source_type"],
            "manifest_sha256": report["manifest_sha256"],
            "config_sha256": report["config_sha256"],
        }
    path = out_root / "external_count_sources.json"
    path.write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sample_ids_for(accession: str, entries: list[dict[str, Any]]) -> list[str]:
    for entry in entries:
        if str(entry["accession"]).upper() == accession:
            return [str(x).upper() for x in entry["samples"]]
    raise KeyError(accession)


def pipeline(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo).resolve()
    config = load_config(repo_root)
    selected = {x.upper() for x in args.accessions.split(",") if x.strip()}
    allowed = {x.upper() for x in config["accessions"]}
    invalid = sorted(selected - allowed)
    if invalid:
        die(f"selected accessions are not part of the four-cohort rescue lock: {invalid}")
    if not selected:
        die("no accessions selected")
    out_root = Path(args.work_dir).resolve()
    entries = manifest_entries(repo_root, selected)
    manifest_tsv = out_root / "metadata" / "locked_run_manifest.tsv"

    if args.step == "resolve":
        resolve(repo_root, out_root, selected)
        return

    if not manifest_tsv.exists():
        resolve(repo_root, out_root, selected)

    if args.step == "download":
        download_fastq(manifest_tsv, out_root, config)
        write_done(out_root / "download.done.json", {"run_manifest_sha256": sha256(manifest_tsv)})
        return

    if args.step == "merge":
        for entry in entries:
            accession = str(entry["accession"]).upper()
            for gsm in sample_ids_for(accession, entries):
                concatenate_sample(manifest_tsv, out_root, accession, gsm)
        return

    if args.step == "qc":
        for entry in entries:
            accession = str(entry["accession"]).upper()
            for gsm in sample_ids_for(accession, entries):
                r1, r2, _ = concatenate_sample(manifest_tsv, out_root, accession, gsm)
                run_fastp(r1, r2, out_root, accession, gsm, config)
        return

    if args.step in {"align", "count", "all"}:
        idx, _, gtf = reference(repo_root, out_root, config, args.threads)
        for entry in entries:
            accession = str(entry["accession"]).upper()
            for gsm in sample_ids_for(accession, entries):
                r1, r2, layout = concatenate_sample(manifest_tsv, out_root, accession, gsm)
                q1, q2 = run_fastp(r1, r2, out_root, accession, gsm, config)
                bam = align(q1, q2, out_root, accession, gsm, idx, args.threads)
                if args.step in {"count", "all"}:
                    feature_count(
                        bam,
                        gtf,
                        out_root,
                        accession,
                        gsm,
                        layout == "PAIRED",
                        args.threads,
                        config,
                    )
        if args.step == "align":
            return

    if args.step in {"count", "all"}:
        # Count-only requires BAMs from a previous alignment run; don't rerun alignment.
        idx, _, gtf = reference(repo_root, out_root, config, args.threads)
        for entry in entries:
            accession = str(entry["accession"]).upper()
            for gsm in sample_ids_for(accession, entries):
                bam = out_root / "alignments" / accession / gsm / f"{gsm}.Aligned.sortedByCoord.out.bam"
                if not bam.exists():
                    die(f"{accession}/{gsm}: BAM missing before count: {bam}")
                rows = raw_rows(manifest_tsv, accession, gsm)
                if not rows:
                    die(f"{accession}/{gsm}: no run metadata")
                layout = rows[0]["library_layout"].upper()
                feature_count(
                    bam,
                    gtf,
                    out_root,
                    accession,
                    gsm,
                    layout == "PAIRED",
                    args.threads,
                    config,
                )

    if args.step in {"validate", "all", "count"}:
        for entry in entries:
            accession = str(entry["accession"]).upper()
            sample_ids = sample_ids_for(accession, entries)
            matrix = out_root / "matrices" / f"{accession}.raw_counts.tsv.gz"
            if not matrix.exists():
                matrix = merge_counts(out_root, accession, sample_ids)
            validate_matrix(matrix, repo_root, accession, sample_ids, out_root)
        write_source_registry(out_root, sorted(selected))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="CardiLearn repository root")
    parser.add_argument("--work-dir", default="./step3_reprocessed", help="pipeline work directory")
    parser.add_argument(
        "--accessions",
        default="GSE232259,GSE52313,GSE186875,GSE308783",
        help="comma-separated rescue accessions",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--step",
        required=True,
        choices=["resolve", "download", "merge", "qc", "align", "count", "validate", "all"],
    )
    return parser


if __name__ == "__main__":
    main_args = build_parser().parse_args()
    pipeline(main_args)
