import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "source_rescue_audit.py"
SPEC = importlib.util.spec_from_file_location("source_rescue_audit", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_extract_accessions():
    text = "SRA: SRX20289525 and SRR123456; GSE232259"
    assert audit.extract_accessions(text, audit.SRX_RE) == ["SRX20289525"]
    assert audit.extract_accessions(text, audit.SRR_RE) == ["SRR123456"]


def test_series_root():
    assert audit.series_root("GSE232259").endswith("/series/GSE232nnn/GSE232259/")


def test_geo_candidate_files_separates_normalized():
    records = [{
        "geo_accession": "GSM1",
        "supplementary_file": [
            "counts_raw.txt.gz",
            "expression_FPKM.txt.gz",
            "normalized_counts.txt.gz",
        ],
    }]
    candidates, rejected = audit.geo_candidate_files(records, "GSE1")
    assert candidates == [
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE0nnn/GSE1/suppl/counts_raw.txt.gz"
    ]
    assert len(rejected) == 2


def test_rank_recommends_strict_candidate():
    strict = audit.SourceEvidence(
        source_type="geo_submitter_raw_counts",
        status="candidate",
        accession="GSE1",
    )
    derived = audit.SourceEvidence(
        source_type="recount3_derived_read_counts",
        status="candidate",
        accession="GSE1",
    )
    result = audit.rank_recommendation([derived, strict])
    assert result["recommended_action"] == "validate_strict_candidate"
    assert result["source_type"] == "geo_submitter_raw_counts"


def test_rank_does_not_promote_archs4():
    archs4 = audit.SourceEvidence(
        source_type="archs4_kallisto_rounded",
        status="available",
        accession="GSE1",
    )
    result = audit.rank_recommendation([archs4])
    assert result["recommended_action"] == "do_not_promote_archs4"


def test_rank_falls_back_to_sra():
    result = audit.rank_recommendation([
        audit.SourceEvidence(
            source_type="archs4_kallisto_rounded",
            status="not_found",
            accession="GSE1",
        )
    ])
    assert result["recommended_action"] == "reprocess_sra"


def test_ena_tsv_report_parser(monkeypatch):
    tsv = (
        "study_accession\texperiment_accession\trun_accession\tread_count\tbase_count\tfastq_ftp\tfastq_bytes\n"
        "SRP1\tSRX1\tSRR1\t100\t10000\tftp://example/SRR1.fastq.gz\t1234\n"
    )
    monkeypatch.setattr(audit, "http_text", lambda url, timeout=30: tsv)
    rows = audit.ena_run_report("SRX1")
    assert rows[0]["run_accession"] == "SRR1"
    assert rows[0]["study_accession"] == "SRP1"


def test_archs4_live_series_lookup(monkeypatch):
    html = "<html>GSM1 GSM2</html>"
    monkeypatch.setattr(audit, "http_text", lambda url, timeout=30: html)
    evidence = audit.archs4_evidence("GSE1", ["GSM1", "GSM2"], None)
    assert evidence.status == "available"
    assert evidence.exact_sample_coverage == 2
    assert evidence.exact_sample_expected == 2


def test_run_audit_writes_json_and_csv(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "eligible_tracks": {
            "strict_biological_replicate_benchmark": [{
                "accession": "GSE1",
                "samples": ["GSM1"],
            }]
        }
    }))

    soft = {
        "GSM1": {
            "geo_accession": "GSM1",
            "title": "sham sample",
            "relation": ["SRA: SRX1"],
            "supplementary_file": ["counts_raw.txt.gz"],
        }
    }

    monkeypatch.setattr(audit, "geo_metadata", lambda accession: (soft, "https://example/soft"))
    monkeypatch.setattr(audit, "ena_run_report", lambda srx: [{
        "study_accession": "SRP1",
        "experiment_accession": srx,
        "run_accession": "SRR1",
        "read_count": "100",
        "base_count": "10000",
        "fastq_bytes": "1000",
    }])
    monkeypatch.setattr(audit, "atlas_search", lambda gse: {"hits": []})
    monkeypatch.setattr(
        audit,
        "http_text",
        lambda url, timeout=30: "<html>GSM1</html>" if "archs4/series" in url else "",
    )
    monkeypatch.setattr(audit, "http_status", lambda url: 404)

    output = tmp_path / "report.json"
    payload = audit.run_audit(manifest, output)

    assert output.exists()
    assert output.with_suffix(".csv").exists()
    assert payload["accessions"]["GSE1"]["srr_ids"] == ["SRR1"]
    assert payload["accessions"]["GSE1"]["recommendation"]["recommended_action"] == "validate_strict_candidate"
