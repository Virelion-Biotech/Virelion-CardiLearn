"""Reproducible acquisition helpers for public NCBI GEO datasets.

Raw source data are never committed to this repository. The helpers download GEO
family metadata and optional supplementary archives into a user-selected cache,
then record checksums and source URLs for provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import re
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class GeoResource:
    accession: str
    kind: str
    url: str
    path: str
    sha256: str
    size_bytes: int


def _series_bucket(accession: str) -> str:
    match = re.fullmatch(r"GSE(\d+)", accession.upper())
    if not match:
        raise ValueError(f"invalid GEO series accession: {accession}")
    number = int(match.group(1))
    return f"GSE{number // 1000}nnn"


def geo_urls(accession: str) -> dict[str, str]:
    acc = accession.upper()
    bucket = _series_bucket(acc)
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{bucket}/{acc}"
    return {
        "family_soft": f"{base}/soft/{acc}_family.soft.gz",
        "raw_tar": f"{base}/suppl/{acc}_RAW.tar",
    }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resource(
    accession: str,
    kind: str,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    timeout: int = 120,
) -> GeoResource:
    urls = geo_urls(accession)
    if kind not in urls:
        raise ValueError(f"unsupported GEO resource kind: {kind}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    filename = Path(urls[kind]).name
    target = output / filename
    if not target.exists() or overwrite:
        request = Request(urls[kind], headers={"User-Agent": "Virelion-CardiLearn/0.3"})
        with urlopen(request, timeout=timeout) as response, target.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    resource = GeoResource(
        accession=accession.upper(),
        kind=kind,
        url=urls[kind],
        path=str(target),
        sha256=sha256_file(target),
        size_bytes=target.stat().st_size,
    )
    manifest = output / f"{accession.upper()}_{kind}.json"
    manifest.write_text(json.dumps(asdict(resource), indent=2, sort_keys=True), encoding="utf-8")
    return resource
