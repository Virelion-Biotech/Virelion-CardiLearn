"""Download explicitly declared GEO supplementary processed files.

This acquisition layer is intentionally narrow: it consumes supplementary-file
URLs recorded by NCBI GEO in a local metadata manifest, downloads processed
files only, and writes SHA256 provenance. It never downloads SRA sequencing
archives and never commits downloaded data to the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

ALLOWED_HOSTS = {"ftp.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"}
SHA256_HEX_LENGTH = 64


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError("processed-data URLs must use HTTPS and an NCBI host")
    if not parsed.path:
        raise ValueError("processed-data URL must contain a path")


def validate_expected_hash(value: object) -> str | None:
    if value in (None, ""):
        return None
    expected = str(value).lower()
    if len(expected) != SHA256_HEX_LENGTH or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("sha256 must be a 64-character hexadecimal digest")
    return expected


def download(url: str, destination: Path, max_bytes: int) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Virelion-CardiLearn/0.3"})
    total = 0
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                output.close()
                destination.unlink(missing_ok=True)
                raise ValueError(f"download exceeds max-bytes guard: {destination}")
            output.write(chunk)
    return sha256_file(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download declared GEO processed files")
    parser.add_argument("--manifest", required=True, help="JSON manifest containing a files list")
    parser.add_argument("--output", required=True, help="Directory outside the Git repository")
    parser.add_argument("--max-bytes", type=int, default=5_000_000_000)
    parser.add_argument("--max-total-bytes", type=int, default=20_000_000_000)
    args = parser.parse_args()
    if args.max_bytes <= 0 or args.max_total_bytes <= 0 or args.max_bytes > args.max_total_bytes:
        raise ValueError("byte limits must be positive and max-bytes <= max-total-bytes")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest must contain a non-empty 'files' list")

    root = Path(args.output).resolve()
    records = []
    total_bytes = 0
    seen_paths: set[Path] = set()
    for item in files:
        if not isinstance(item, dict) or not item.get("url") or not item.get("relative_path"):
            raise ValueError("each file entry requires url and relative_path")
        url = str(item["url"])
        validate_source_url(url)
        relative = Path(str(item["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("relative_path must remain inside the output directory")
        if relative in seen_paths:
            raise ValueError(f"duplicate relative_path: {relative}")
        seen_paths.add(relative)
        destination = root / relative
        expected = validate_expected_hash(item.get("sha256"))
        if destination.exists() and destination.stat().st_size > args.max_bytes:
            raise ValueError(f"existing file exceeds max-bytes guard: {destination}")
        if expected and destination.exists():
            actual = sha256_file(destination)
            if actual != expected:
                raise ValueError(f"existing file hash mismatch: {destination}")
            digest = actual
        else:
            digest = download(url, destination, args.max_bytes)
        size = destination.stat().st_size
        if expected and digest != expected:
            destination.unlink(missing_ok=True)
            raise ValueError(f"download hash mismatch: {destination}")
        total_bytes += size
        if total_bytes > args.max_total_bytes:
            raise ValueError("total acquisition exceeds max-total-bytes guard")
        records.append(
            {
                "accession": str(item.get("accession", "")),
                "url": url,
                "relative_path": str(relative),
                "bytes": size,
                "sha256": digest,
            }
        )

    provenance = {
        "schema_version": "1.1",
        "source_manifest": str(Path(args.manifest).resolve()),
        "processed_data_only": True,
        "raw_sra_downloaded": False,
        "total_bytes": total_bytes,
        "files": records,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "acquisition_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
