"""Download explicitly declared GEO supplementary processed files.

This acquisition layer is intentionally narrow: it consumes the supplementary-file
URLs recorded by NCBI GEO in a local metadata manifest, downloads processed files only,
and writes SHA256 provenance. It never downloads SRA sequencing archives and never
commits downloaded data to the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Virelion-CardiLearn/0.3"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    return sha256_file(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download declared GEO processed files")
    parser.add_argument("--manifest", required=True, help="JSON manifest containing a files list")
    parser.add_argument("--output", required=True, help="Directory outside the Git repository")
    parser.add_argument("--max-bytes", type=int, default=5_000_000_000)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest must contain a non-empty 'files' list")

    root = Path(args.output).resolve()
    records = []
    for item in files:
        if not isinstance(item, dict) or not item.get("url") or not item.get("relative_path"):
            raise ValueError("each file entry requires url and relative_path")
        relative = Path(str(item["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("relative_path must remain inside the output directory")
        destination = root / relative
        if destination.exists() and destination.stat().st_size > args.max_bytes:
            raise ValueError(f"existing file exceeds max-bytes guard: {destination}")
        expected = item.get("sha256")
        if expected and destination.exists():
            actual = sha256_file(destination)
            if actual != str(expected):
                raise ValueError(f"existing file hash mismatch: {destination}")
            digest = actual
        else:
            digest = download(str(item["url"]), destination)
        if destination.stat().st_size > args.max_bytes:
            destination.unlink(missing_ok=True)
            raise ValueError(f"download exceeds max-bytes guard: {destination}")
        if expected and digest != str(expected):
            destination.unlink(missing_ok=True)
            raise ValueError(f"download hash mismatch: {destination}")
        records.append(
            {
                "accession": str(item.get("accession", "")),
                "url": str(item["url"]),
                "relative_path": str(relative),
                "bytes": destination.stat().st_size,
                "sha256": digest,
            }
        )

    provenance = {
        "schema_version": "1.0",
        "source_manifest": str(Path(args.manifest).resolve()),
        "processed_data_only": True,
        "raw_sra_downloaded": False,
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
