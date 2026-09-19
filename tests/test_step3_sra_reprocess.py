import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "step3_sra_reprocess.py"
SPEC = importlib.util.spec_from_file_location("step3_sra_reprocess", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_ena_rows_parses_tsv(monkeypatch):
    body = (
        "study_accession\texperiment_accession\trun_accession\tsample_accession\t"
        "library_layout\tlibrary_strategy\tinstrument_platform\tinstrument_model\tread_count\t"
        "base_count\tfastq_ftp\tfastq_md5\tfastq_bytes\n"
        "SRP1\tSRX1\tSRR1\tERS1\tPAIRED\tRNA-Seq\tILLUMINA\tNovaSeq\t100\t15000\t"
        "ftp://ftp.sra.ebi.ac.uk/vol1/SRR1_1.fastq.gz;ftp://ftp.sra.ebi.ac.uk/vol1/SRR1_2.fastq.gz\t"
        "abc;def\t1000;2000\n"
    )
    monkeypatch.setattr(mod, "http_text", lambda url, timeout=120: body)
    rows = mod.ena_rows("SRX1")
    assert len(rows) == 1
    assert rows[0]["run_accession"] == "SRR1"
    assert rows[0]["library_layout"] == "PAIRED"
    assert rows[0]["fastq_ftp"].count(";") == 1


def test_extract_srx_is_unique_and_sorted():
    values = mod.extract_srx([
        "SRA: https://www.ncbi.nlm.nih.gov/sra?term=SRX2",
        "SRA: https://www.ncbi.nlm.nih.gov/sra?term=SRX1; SRX2",
    ])
    assert values == ["SRX1", "SRX2"]


def test_fastq_dest_uses_ena_filename(tmp_path):
    row = {
        "accession": "GSE1",
        "gsm": "GSM1",
        "srr": "SRR1",
        "file_index": "1",
        "fastq_ftp": "ftp://ftp.sra.ebi.ac.uk/vol1/SRR1_1.fastq.gz",
    }
    out = mod.fastq_dest(row, tmp_path)
    assert out.name == "SRR1_1.fastq.gz"
    assert out.parts[-3:] == ("GSM1", "SRR1", "SRR1_1.fastq.gz")


def test_parse_featurecounts_rejects_non_integer(tmp_path):
    path = tmp_path / "GSM1.featureCounts.txt"
    path.write_text(
        "# Program: featureCounts\n"
        "Geneid\tChr\tStart\tEnd\tStrand\tLength\tGSM1.bam\n"
        "ENSMUSG1\t1\t1\t10\t+\t10\t42\n"
        "ENSMUSG2\t1\t20\t30\t-\t11\t0\n"
    )
    sample, counts = mod.parse_featurecounts(path)
    assert sample == "GSM1"
    assert counts == {"ENSMUSG1": 42, "ENSMUSG2": 0}


def test_parse_featurecounts_fails_fractional(tmp_path):
    path = tmp_path / "GSM1.featureCounts.txt"
    path.write_text(
        "Geneid\tChr\tStart\tEnd\tStrand\tLength\tGSM1.bam\n"
        "ENSMUSG1\t1\t1\t10\t+\t10\t1.5\n"
    )
    try:
        mod.parse_featurecounts(path)
    except SystemExit as exc:
        assert "non-integer" in str(exc)
    else:
        raise AssertionError("fractional counts were accepted")


def test_config_is_locked_to_four_rescue_accessions():
    config = mod.load_config(Path(__file__).resolve().parents[1])
    assert config["accessions"] == [
        "GSE232259",
        "GSE52313",
        "GSE186875",
        "GSE308783",
    ]
    assert config["reference"]["assembly"] == "GRCm39"
    assert config["reference"]["ensembl_release"] == 112
    assert config["counting"]["strand"] == 0
    assert config["counting"]["fractional"] is False


def test_feature_counts_source_registry_requires_hashes(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    (root / "validation").mkdir()
    (root / "validation" / "GSE1.json").write_text(
        '{"source_type":"sra_reprocessed_raw_counts","source":"matrix.tsv.gz",'
        '"sha256":"' + ("a" * 64) + '","manifest_sha256":"' + ("b" * 64) +
        '","config_sha256":"' + ("c" * 64) + '"}'
    )
    path = mod.write_source_registry(root, ["GSE1"])
    payload = mod.load_json(path)
    assert payload["GSE1"]["sha256"] == "a" * 64
    assert payload["GSE1"]["source_type"] == "sra_reprocessed_raw_counts"
