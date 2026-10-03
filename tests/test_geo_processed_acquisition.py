from __future__ import annotations

import json

import pytest

from scripts.fetch_geo_supplementary import validate_expected_hash, validate_source_url


def test_ncbi_https_url_is_accepted() -> None:
    validate_source_url("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM0000/GSM0000/suppl/file.mtx.gz")


def test_non_ncbi_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="NCBI host"):
        validate_source_url("https://example.org/file.mtx.gz")


def test_http_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        validate_source_url("http://ftp.ncbi.nlm.nih.gov/file.mtx.gz")


def test_invalid_sha256_is_rejected() -> None:
    with pytest.raises(ValueError, match="64-character"):
        validate_expected_hash("not-a-hash")


def test_valid_sha256_is_normalized() -> None:
    digest = "A" * 64
    assert validate_expected_hash(digest) == digest.lower()


def test_empty_sha256_is_allowed() -> None:
    assert validate_expected_hash("") is None
    assert validate_expected_hash(None) is None


def test_manifest_is_valid_json() -> None:
    from pathlib import Path

    payload = json.loads(Path("configs/geo_gse153480_processed_manifest.json").read_text())
    assert payload["accession"] == "GSE153480"
    assert len(payload["files"]) == 24
