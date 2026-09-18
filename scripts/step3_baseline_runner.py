from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import numpy as np
import pandas as pd
import requests
import yaml


class Step3Halt(RuntimeError):
    pass


COMMIT = "e57b810a2680a1c8fdd7ea6b64ee782f0d744a43"
MANIFEST_SHA = "614807b68606ef6248913becd90c9b45240401f6a3a8c511c03dde41846555c1"
BENCHMARK_SHA = "971202aa93b482f167cab376ea1a3f5ee1609137b5269a1f6a546ae357e32c3e"
REPO_URL = "https://github.com/Virelion-Biotech/Virelion-CardiLearn.git"
WORK = Path("/content/CardiLearn_step3_work")
REPO = WORK / "repo"
SRC = WORK / "sources"
EXT = WORK / "extracted"
CACHE = WORK / "_pycache"
REPORT = REPO / "reports" / "baselines_v1.json"
AUDIT = WORK / "source_audits"
ACCESSIONS = ["GSE183272", "GSE236374", "GSE232259", "GSE52313", "GSE186875", "GSE308783"]
TRAIN = {"GSE183272", "GSE232259", "GSE52313", "GSE186875"}
VALIDATION = {"GSE236374"}
TEST = {"GSE308783"}
SEED = 42
N_GENES = 5000
N_BOOT = 2000
N_PERM = 1000
TIMEOUT = (20, 120)
MAX_BYTES = 2_000_000_000
GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo"

GSM_RE = re.compile(r"(?<![A-Za-z0-9])(GSM\d+)(?!\d)", re.I)
ENS_RE = re.compile(r"^ENSMUSG\d+$", re.I)
MI_RE = re.compile(r"(?<![A-Za-z0-9])(?:mi\d*|m[.\s_-]*i|myocardial infarction|infarct)(?![A-Za-z0-9])", re.I)
SHAM_RE = re.compile(r"(?<![A-Za-z0-9])sham(?![A-Za-z0-9])|sham[-_ ]?operated|vehicle\s+control", re.I)
FORBIDDEN_RE = re.compile(r"(?<![A-Za-z0-9])(?:fpkm|tpm|cpm|rpkm|normalized|normalised|log2|log1p)(?![A-Za-z0-9])", re.I)
COUNT_RE = re.compile(r"^(?:count|counts|genecount|gene_count|readcount|read_count)$", re.I)


def halt(message: str) -> None:
    raise Step3Halt(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_checked(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    e = os.environ.copy()
    if env:
        e.update(env)
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e["PYTHONPYCACHEPREFIX"] = str(CACHE)
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=e, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode:
        halt(f"Subprocess failure.\nCommand: {' '.join(cmd)}\nExit code: {p.returncode}\nOutput:\n{p.stdout}")
    return p.stdout


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, timeout=TIMEOUT, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            length = r.headers.get("Content-Length")
            if length and int(length) > MAX_BYTES:
                halt(f"External source exceeds safety limit.\nURL: {url}\nContent-Length: {length}")
            total = 0
            with path.open("wb") as out:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        total += len(chunk)
                        if total > MAX_BYTES:
                            halt(f"External source exceeded safety limit.\nURL: {url}")
                        out.write(chunk)
    except Step3Halt:
        raise
    except Exception as exc:
        halt(f"External data access failure.\nURL: {url}\nError: {type(exc).__name__}: {exc}")


def series_root(acc: str) -> str:
    n = int(acc[3:])
    return f"{GEO_FTP}/series/GSE{n // 1000}nnn/{acc}/"


def family_soft_url(acc: str) -> str:
    return f"{series_root(acc)}soft/{acc}_family.soft.gz"


def parse_soft(text: str) -> list[dict[str, object]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("^SAMPLE = "):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)

    records = []
    for block in blocks:
        if "=" not in block[0]:
            halt(f"Malformed GEO SAMPLE header: {block[0]}")
        rec: dict[str, object] = {"geo_accession": block[0].split("=", 1)[1].strip()}
        for line in block[1:]:
            if not line.startswith("!Sample_") or " = " not in line:
                continue
            key, value = line.split(" = ", 1)
            key = key[len("!Sample_"):]
            value = value.strip().replace('\\"', '"')
            if key in {"characteristics_ch1", "supplementary_file", "relation", "data_processing"}:
                rec.setdefault(key, []).append(value)
            elif key not in rec:
                rec[key] = value
        records.append(rec)
    return records


def _text(rec: dict[str, object], key: str) -> str:
    value = rec.get(key, [])
    if isinstance(value, list):
        return " | ".join(str(x) for x in value).lower()
    return str(value).lower()


def classify_condition(rec: dict[str, object]) -> str | None:
    for value in (_text(rec, "characteristics_ch1"), _text(rec, "title"), _text(rec, "source_name_ch1")):
        if not value:
            continue
        mi = bool(MI_RE.search(value))
        sham = bool(SHAM_RE.search(value))
        if mi and sham:
            continue
        if mi:
            return "MI"
        if sham:
            return "sham"
    return None


def listing_urls(url: str) -> list[str]:
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception as exc:
        halt(f"GEO supplementary directory access failure.\nURL: {url}\nError: {type(exc).__name__}: {exc}")
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', r.text, re.I)
    return sorted({
        urljoin(url, unquote(h))
        for h in hrefs
        if not h.startswith(("../", "./", "#"))
    })


def source_urls(acc: str, samples: list[dict[str, object]]) -> list[str]:
    expected = {str(x["geo_accession"]).upper() for x in samples}
    explicit: set[str] = set()
    per_gsm: dict[str, str] = {}

    for rec in samples:
        gsm = str(rec["geo_accession"]).upper()
        values = rec.get("supplementary_file", [])
        if not isinstance(values, list):
            continue
        for value in values:
            u = str(value).strip()
            if u in {"", "NONE", "null", "NA"}:
                continue
            if u.startswith("ftp://"):
                u = "https://" + u[6:]
            elif not u.startswith(("http://", "https://")):
                u = urljoin(f"{series_root(acc)}suppl/", u.lstrip("/"))
            explicit.add(u)
            ids = GSM_RE.findall(Path(urlparse(u).path).name)
            if len(ids) == 1 and ids[0].upper() == gsm:
                per_gsm[gsm] = u

    if expected and expected.issubset(per_gsm):
        return [per_gsm[g] for g in sorted(expected)]

    discovered = set(explicit)
    discovered.update(listing_urls(f"{series_root(acc)}suppl/"))
    return sorted(discovered, key=candidate_score, reverse=True)


def candidate_score(url: str) -> int:
    name = Path(urlparse(url).path).name.lower()
    if not name or "filelist" in name:
        return -10_000
    score = 0
    if GSM_RE.search(name):
        score += 100
    for token in ("raw", "count", "counts", "featurecount", "genecount", "readcount"):
        if token in name:
            score += 10
    if FORBIDDEN_RE.search(name):
        score -= 500
    if name.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
        score += 2
    return score


def usable_file(name: str) -> bool:
    return name.lower().endswith((".txt", ".tsv", ".csv", ".tab", ".gz", ".tar", ".tar.gz", ".tgz", ".zip"))


def safe_extract(path: Path, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    lower = path.name.lower()
    try:
        if lower.endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                members = z.infolist()
                for info in members:
                    if info.is_dir():
                        continue
                    target = (outdir / info.filename).resolve()
                    if not str(target).startswith(str(outdir.resolve()) + os.sep):
                        halt(f"Archive path traversal detected: {info.filename}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                    extracted.append(target)
        else:
            mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(path, mode) as t:
                for member in t.getmembers():
                    if not member.isfile():
                        continue
                    target = (outdir / member.name).resolve()
                    if not str(target).startswith(str(outdir.resolve()) + os.sep):
                        halt(f"Archive path traversal detected: {member.name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = t.extractfile(member)
                    if src is None:
                        continue
                    with src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                    extracted.append(target)
    except Step3Halt:
        raise
    except Exception as exc:
        halt(f"Archive extraction failure.\nArchive: {path}\nError: {type(exc).__name__}: {exc}")
    return extracted


def open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="strict")
    return path.open("rt", encoding="utf-8", errors="strict")


def read_table(path: Path) -> pd.DataFrame:
    with open_text(path) as fh:
        preview = fh.read(16384)
    lines = [x for x in preview.splitlines() if x.strip() and not x.lstrip().startswith("#")]
    if not lines:
        raise ValueError("empty text file")
    first = lines[0]
    separators = ["\t", ",", r"\s+"]
    if "\t" not in first:
        separators = [",", r"\s+", "\t"] if "," in first else [r"\s+", "\t", ","]
    last_error = None
    for sep in separators:
        try:
            with open_text(path) as fh:
                df = pd.read_csv(fh, sep=sep, comment="#", low_memory=False)
            if df.shape[1] >= 2:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception as exc:
            last_error = exc
    raise ValueError(f"unable to parse tabular file: {last_error}")


def integer_values(series: pd.Series) -> tuple[np.ndarray, bool]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    ok = np.isfinite(values).all() and (values >= 0).all()
    if not ok:
        return values, False
    return values, bool(np.isclose(values, np.rint(values), rtol=0.0, atol=0.0).all())


def parse_count_file(path: Path, expected: list[str]) -> dict[str, pd.Series]:
    df = read_table(path)
    expected_upper = {x.upper() for x in expected}
    gene_col = df.columns[0]

    # Matrix path: sample/GSM identifiers are present in column names.
    matrix_cols: dict[str, str] = {}
    for col in df.columns[1:]:
        ids = GSM_RE.findall(str(col))
        for gsm in ids:
            if gsm.upper() in expected_upper:
                matrix_cols[gsm.upper()] = col
    if matrix_cols:
        out: dict[str, pd.Series] = {}
        genes = df[gene_col].astype(str).str.strip()
        for gsm, col in matrix_cols.items():
            values, integer = integer_values(df[col])
            if not integer:
                raise ValueError(f"{col} is not raw integer-like counts")
            out[gsm] = pd.Series(values.astype(np.int64), index=genes, name=gsm)
        return out

    # Single-sample path: GSM is in filename, exactly matching one manifest unit.
    filename_ids = [x.upper() for x in GSM_RE.findall(path.name) if x.upper() in expected_upper]
    if len(filename_ids) != 1:
        raise ValueError("no unique manifest GSM in filename and no manifest GSM columns")

    gsm = filename_ids[0]
    named_count_cols = [c for c in df.columns[1:] if COUNT_RE.fullmatch(str(c).strip())]
    if len(named_count_cols) == 0:
        numeric_cols = []
        for col in df.columns[1:]:
            _, integer = integer_values(df[col])
            if integer:
                numeric_cols.append(col)
        if len(numeric_cols) == 1:
            named_count_cols = numeric_cols
    if len(named_count_cols) != 1:
        raise ValueError(f"could not uniquely identify count column: {list(df.columns)}")
    values, integer = integer_values(df[named_count_cols[0]])
    if not integer:
        raise ValueError("single-sample count column is not raw integer-like")
    genes = df[gene_col].astype(str).str.strip()
    return {gsm: pd.Series(values.astype(np.int64), index=genes, name=gsm)}


def normalize_genes(df: pd.DataFrame) -> pd.DataFrame:
    genes = df["gene_id"].astype(str).str.strip().str.replace(r"\.\d+$", "", regex=True)
    keep = genes.str.match(ENS_RE)
    n = int(keep.sum())
    if n < 100:
        halt(f"Gene identifier gate failed: only {n} stable ENSMUSG identifiers")
    df = df.loc[keep].copy()
    df["gene_id"] = genes.loc[keep]
    if df["gene_id"].duplicated().any():
        halt("Duplicate stable mouse gene identifiers detected")
    return df


def combine_series(parsed: dict[str, pd.Series], expected: list[str]) -> pd.DataFrame:
    if {x.upper() for x in parsed} != {x.upper() for x in expected}:
        missing = sorted({x.upper() for x in expected} - {x.upper() for x in parsed})
        extra = sorted({x.upper() for x in parsed} - {x.upper() for x in expected})
        halt(f"Raw-source sample coverage mismatch. Missing={missing}; Extra={extra}")
    first = next(iter(parsed.values()))
    if first.index.duplicated().any():
        halt("Duplicate gene identifiers in raw source")
    for series in parsed.values():
        if series.index.duplicated().any() or set(series.index) != set(first.index):
            halt("Per-sample raw count files do not contain identical gene sets")
    matrix = pd.concat([parsed[g.upper()] for g in expected], axis=1)
    matrix.columns = expected
    return normalize_genes(matrix.reset_index().rename(columns={"index": "gene_id"}))


def acquire_counts(acc: str, rec: dict[str, object], samples: list[dict[str, object]]) -> tuple[pd.DataFrame, dict[str, object]]:
    expected = list(rec["samples"])
    urls = source_urls(acc, samples)
    if not urls:
        halt(f"No GEO supplementary expression sources discovered for {acc}")

    parsed_samples: dict[str, pd.Series] = {}
    audit: list[dict[str, str]] = []
    downloaded: list[dict[str, str]] = []

    for idx, url in enumerate(urls):
        filename = Path(urlparse(url).path).name
        if not filename or not usable_file(filename) or "filelist" in filename.lower():
            audit.append({"url": url, "status": "skipped", "reason": "not an expression candidate"})
            continue

        local = SRC / acc / f"{idx:03d}_{filename}"
        download(url, local)
        downloaded.append({"url": url, "path": str(local), "sha256": sha256_file(local)})

        files = [local]
        if local.name.lower().endswith((".tar", ".tar.gz", ".tgz", ".zip")):
            files = safe_extract(local, EXT / acc)

        for file in files:
            if not file.is_file() or not usable_file(file.name) or "filelist" in file.name.lower():
                continue
            if FORBIDDEN_RE.search(file.name.lower()):
                audit.append({"url": url, "status": "rejected", "reason": f"normalized-scale filename: {file.name}"})
                continue
            try:
                parsed = parse_count_file(file, expected)
                for gsm, series in parsed.items():
                    if gsm.upper() in parsed_samples:
                        halt(f"Multiple independent source files map to GSM {gsm} in {acc}; mapping is ambiguous")
                    parsed_samples[gsm.upper()] = series
                audit.append({"url": url, "status": "accepted", "reason": file.name})

                if {x.upper() for x in parsed_samples} == {x.upper() for x in expected}:
                    matrix = combine_series(parsed_samples, expected)
                    return matrix, {
                        "source_contract": "raw_integer_like_counts",
                        "accepted_source_file": str(file),
                        "accepted_source_sha256": sha256_file(file),
                        "downloaded_sources": downloaded,
                        "candidate_audit": audit,
                    }
            except Exception as exc:
                audit.append({"url": url, "status": "rejected", "reason": f"{type(exc).__name__}: {exc}"})

    audit_path = AUDIT / f"{acc}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "accession": acc,
        "expected_samples": expected,
        "candidate_audit": audit,
        "sra_relations": [
            str(v)
            for r in samples
            for v in (r.get("relation", []) if isinstance(r.get("relation", []), list) else [])
            if "sra" in str(v).lower()
        ],
    }, indent=2, sort_keys=True), encoding="utf-8")
    halt(
        "Raw integer-like count source contract failed.\n"
        f"Accession: {acc}\n"
        f"Candidate audit: {audit_path}\n"
        + json.dumps(audit, indent=2)
    )


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
    pred = (p >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "brier_score": float(brier_score_loss(y, p)),
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p <= hi) if hi == 1 else (p < hi))
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def bootstrap(y: np.ndarray, p: np.ndarray) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    zero = np.flatnonzero(y == 0)
    one = np.flatnonzero(y == 1)
    store = {k: [] for k in ("auroc", "auprc", "balanced_accuracy", "f1_macro", "brier_score")}
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(zero, len(zero), replace=True), rng.choice(one, len(one), replace=True)])
        m = metrics(y[idx], p[idx])
        for k, v in m.items():
            store[k].append(v)
    obs = metrics(y, p)
    return {
        k: {
            "estimate": obs[k],
            "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
            "n_bootstrap": N_BOOT,
        }
        for k, v in store.items()
    }


def permutation_p(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED)
    observed = roc_auc_score(y, p)
    shuffled = y.copy()
    ge = 0
    for _ in range(N_PERM):
        rng.shuffle(shuffled)
        if roc_auc_score(shuffled, p) >= observed:
            ge += 1
    return float((ge + 1) / (N_PERM + 1))


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    for path in (WORK, SRC, EXT, CACHE, AUDIT):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPYCACHEPREFIX"] = str(CACHE)
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(CACHE)

    print("=" * 80)
    print("CARDILEARN STEP 3 — CLEAN START")
    print("=" * 80)
    print("Python:", sys.version.splitlines()[0])

    for pkg in ("numpy", "pandas", "yaml", "sklearn", "scipy", "torch"):
        try:
            mod = __import__(pkg)
            print(f"{pkg}:", getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            print(f"{pkg}: unavailable ({type(exc).__name__}: {exc})")

    try:
        import torch
        print("CUDA available:", bool(torch.cuda.is_available()))
        print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    except Exception:
        print("CUDA available: False")
        print("CUDA device: CPU")

    if REPO.exists():
        shutil.rmtree(REPO)
    run_checked(["git", "clone", "--quiet", REPO_URL, str(REPO)])
    run_checked(["git", "checkout", "--quiet", COMMIT], cwd=REPO)

    head = run_checked(["git", "rev-parse", "HEAD"], cwd=REPO).strip()
    if head != COMMIT:
        halt(f"Repository commit mismatch: {head} != {COMMIT}")
    if run_checked(["git", "status", "--porcelain"], cwd=REPO).strip():
        halt("Fresh checkout is dirty")

    print("\nREPOSITORY CHECKOUT")
    print("Current HEAD:", head)
    print("Expected HEAD:", COMMIT)
    print("Repository commit lock: PASS")
    print("Fresh checkout cleanliness: PASS")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")

    print("\nPRE-FLIGHT REPOSITORY SCRUB")
    for path in sorted(REPO.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            halt(f"In-memory Python syntax scrub failed.\nFile: {path}\nError: {exc}")
    print("In-memory Python syntax scrub: PASS")

    print(run_checked(
        [sys.executable, "-B", "-c", "import cardilearn; import scripts; print('cardilearn import:', cardilearn.__version__); print('scripts import: PASS')"],
        cwd=REPO,
        env=env,
    ).strip())

    pytest = run_checked([sys.executable, "-B", "-m", "pytest", "-q"], cwd=REPO, env=env)
    print(pytest)

    if list(REPO.rglob("__pycache__")) or list(REPO.rglob("*.egg-info")):
        halt("Repository generated forbidden cache/build artifacts")
    if run_checked(["git", "status", "--porcelain"], cwd=REPO).strip():
        halt("Pre-flight worktree is dirty")
    print("Pre-flight worktree cleanliness: PASS")

    manifest_path = REPO / "data" / "manifest.lock.json"
    benchmark_path = REPO / "configs" / "benchmark_v1.lock.yaml"
    manifest_sha = sha256_file(manifest_path)
    benchmark_sha = sha256_file(benchmark_path)

    print("\nLOCK ARTIFACT VERIFICATION")
    print("Approved manifest SHA256:", MANIFEST_SHA)
    print("Current manifest SHA256: ", manifest_sha)
    print("Approved benchmark SHA256:", BENCHMARK_SHA)
    print("Current benchmark SHA256: ", benchmark_sha)

    if manifest_sha != MANIFEST_SHA:
        halt("Manifest SHA256 mismatch against approved artifact")
    if benchmark_sha != BENCHMARK_SHA:
        halt("Benchmark SHA256 mismatch against approved artifact")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))

    strict = manifest["eligible_tracks"]["strict_biological_replicate_benchmark"]
    by_acc = {x["accession"]: x for x in strict}
    if set(by_acc) != set(ACCESSIONS):
        halt("Manifest strict accession membership mismatch")
    if set(benchmark["cohort_inclusion"]["accessions"]) != set(ACCESSIONS):
        halt("Benchmark cohort inclusion mismatch")

    assignment = benchmark["frozen_split"]["assignment"]
    train_fam = set(assignment["train"])
    val_fam = set(assignment["validation"])
    test_fam = set(assignment["test"])

    if train_fam != TRAIN or val_fam != VALIDATION or test_fam != TEST:
        halt("Frozen split does not match the approved benchmark")
    if train_fam & val_fam or train_fam & test_fam or val_fam & test_fam:
        halt("Frozen split families overlap")

    print("Manifest bytes match approved artifact: PASS")
    print("Benchmark bytes match approved artifact: PASS")
    print("\nLOCK STRUCTURE")
    print("Manifest strict accession membership: PASS")
    print("Benchmark cohort inclusion membership: PASS")
    print("Frozen split: PASS")

    metadata: dict[str, list[dict[str, object]]] = {}
    matrices: dict[str, pd.DataFrame] = {}
    provenance: dict[str, object] = {}

    print("\nGEO METADATA RECONCILIATION + RAW COUNT SOURCE AUDIT")

    for acc in ACCESSIONS:
        rec = by_acc[acc]
        soft_url = family_soft_url(acc)
        soft_path = SRC / acc / f"{acc}_family.soft.gz"
        download(soft_url, soft_path)
        try:
            with gzip.open(soft_path, "rt", encoding="utf-8", errors="strict") as fh:
                all_samples = parse_soft(fh.read())
        except Exception as exc:
            halt(f"GEO family SOFT parsing failure for {acc}: {type(exc).__name__}: {exc}")

        sample_map = {str(x["geo_accession"]): x for x in all_samples}
        wanted = list(rec["samples"])
        if set(wanted) - set(sample_map):
            halt(f"GEO metadata missing manifest samples for {acc}: {sorted(set(wanted) - set(sample_map))}")

        selected = [sample_map[gsm] for gsm in wanted]
        conditions = [classify_condition(x) for x in selected]

        if acc == "GSE186875":
            checked = []
            for gsm, condition in zip(wanted, conditions):
                title = str(sample_map[gsm].get("title", "")).lower()
                if "vehicle" not in title:
                    halt(f"GSE186875 selected sample lacks explicit vehicle title evidence: {gsm}")
                if "prednisone" in title:
                    halt(f"GSE186875 selected sample unexpectedly contains prednisone: {gsm}")
                if condition not in {"MI", "sham"}:
                    halt(f"GSE186875 condition unresolved: {gsm}")
                checked.append(condition)
            conditions = checked
        elif set(conditions) != {"MI", "sham"}:
            halt(f"Condition reconciliation mismatch for {acc}: {sorted(set(conditions))}")

        metadata[acc] = [
            {
                "sample_id": gsm,
                "injury": condition,
                "study_family_id": acc,
                "accession": acc,
            }
            for gsm, condition in zip(wanted, conditions)
        ]

        matrix, info = acquire_counts(acc, rec, selected)
        matrices[acc] = matrix.set_index("gene_id")[wanted]
        provenance[acc] = {"family_soft_url": soft_url, **info}

        print(f"{acc}: metadata={len(wanted)}; genes={matrix.shape[0]}; raw count source=PASS")

    meta = pd.DataFrame([x for rows in metadata.values() for x in rows]).set_index("sample_id")

    common = set(next(iter(matrices.values())).index)
    for matrix in matrices.values():
        common &= set(matrix.index)
    common = sorted(common)

    if len(common) < N_GENES:
        halt(f"Common stable mouse gene universe too small: {len(common)} < {N_GENES}")

    counts = pd.concat([matrices[acc].loc[common] for acc in ACCESSIONS], axis=1)
    counts = counts[meta.index]

    train_samples = meta.index[meta["study_family_id"].isin(train_fam)]
    val_samples = meta.index[meta["study_family_id"].isin(val_fam)]
    test_samples = meta.index[meta["study_family_id"].isin(test_fam)]

    variance = counts[train_samples].var(axis=1, ddof=0)
    selected_genes = variance.sort_values(ascending=False).head(N_GENES).index.tolist()
    if len(selected_genes) != N_GENES:
        halt("Training-only variable-gene selection returned incorrect size")

    library_sizes = counts.loc[selected_genes].sum(axis=0)
    if (library_sizes <= 0).any():
        halt(f"Non-positive library size: {library_sizes[library_sizes <= 0].index.tolist()}")

    log_cpm = np.log1p(counts.loc[selected_genes] / library_sizes * 1_000_000.0)

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(log_cpm[train_samples].T)
    Xva = scaler.transform(log_cpm[val_samples].T)
    Xte = scaler.transform(log_cpm[test_samples].T)

    ytr = (meta.loc[train_samples, "injury"] == "MI").astype(int).to_numpy()
    yva = (meta.loc[val_samples, "injury"] == "MI").astype(int).to_numpy()
    yte = (meta.loc[test_samples, "injury"] == "MI").astype(int).to_numpy()

    if any(len(np.unique(y)) != 2 for y in (ytr, yva, yte)):
        halt("At least one benchmark split is single-class")

    predictions: dict[str, np.ndarray] = {}
    models: dict[str, object] = {}

    pca_n = min(32, Xtr.shape[0] - 1, Xtr.shape[1])
    pca = PCA(n_components=pca_n, svd_solver="full", random_state=SEED).fit(Xtr)
    pca_probe = LogisticRegression(max_iter=2000, random_state=SEED).fit(pca.transform(Xtr), ytr)
    p = pca_probe.predict_proba(pca.transform(Xte))[:, 1]
    predictions["pca_linear_probe"] = p.copy()
    models["pca_linear_probe"] = {
        "validation": metrics(yva, pca_probe.predict_proba(pca.transform(Xva))[:, 1]),
        "test": metrics(yte, p),
        "calibration": {"ece": ece(yte, p), "n_bins": 10},
        "pca_components": pca_n,
    }

    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=SEED,
        batch_size=min(16, len(ytr)),
        early_stopping=False,
    ).fit(Xtr, ytr)
    p = mlp.predict_proba(Xte)[:, 1]
    predictions["plain_mlp"] = p.copy()
    models["plain_mlp"] = {
        "validation": metrics(yva, mlp.predict_proba(Xva)[:, 1]),
        "test": metrics(yte, p),
        "calibration": {"ece": ece(yte, p), "n_bins": 10},
        "hidden_layer_sizes": [128],
        "max_iter": 500,
    }

    import torch
    import torch.nn as nn

    class PlainAutoencoder(nn.Module):
        def __init__(self, d: int, z: int):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, z))
            self.decoder = nn.Sequential(nn.Linear(z, 256), nn.ReLU(), nn.Linear(256, d))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = min(64, Xtr.shape[0] - 1)
    ae = PlainAutoencoder(Xtr.shape[1], latent_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    train_tensor = torch.tensor(Xtr, dtype=torch.float32, device=device)

    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(ae(train_tensor), train_tensor)
        loss.backward()
        optimizer.step()

    def encode(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return ae.encoder(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()

    ztr = encode(Xtr)
    zva = encode(Xva)
    zte = encode(Xte)
    ae_probe = LogisticRegression(max_iter=2000, random_state=SEED).fit(ztr, ytr)
    p = ae_probe.predict_proba(zte)[:, 1]
    predictions["plain_autoencoder"] = p.copy()
    models["plain_autoencoder"] = {
        "validation": metrics(yva, ae_probe.predict_proba(zva)[:, 1]),
        "test": metrics(yte, p),
        "calibration": {"ece": ece(yte, p), "n_bins": 10},
        "latent_dim": latent_dim,
        "epochs": 200,
    }

    for name, p in predictions.items():
        models[name]["test_uncertainty"] = bootstrap(yte, p)
        models[name]["test_permutation_pvalue_auroc"] = permutation_p(yte, p)

    report = {
        "schema_version": "1.0",
        "step": 3,
        "artifact": "reports/baselines_v1.json",
        "repository_commit": COMMIT,
        "manifest_sha256": manifest_sha,
        "benchmark_sha256": benchmark_sha,
        "strict_accessions": ACCESSIONS,
        "split": {
            "train_families": sorted(train_fam),
            "validation_families": sorted(val_fam),
            "test_families": sorted(test_fam),
            "n_train_units": len(train_samples),
            "n_validation_units": len(val_samples),
            "n_test_units": len(test_samples),
        },
        "selected_gene_count": len(selected_genes),
        "selected_gene_sha256": hashlib.sha256("\n".join(selected_genes).encode()).hexdigest(),
        "normalization": "log1p(CPM); StandardScaler fit on training families only",
        "low_sample_warning": len(test_samples) < 10,
        "source_provenance": provenance,
        "models": models,
        "interpretation_boundary": benchmark["primary_task"]["interpretation_boundary"],
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if run_checked(["git", "status", "--porcelain"], cwd=REPO).strip() != "?? reports/baselines_v1.json":
        halt("Unexpected repository modification after Step 3")

    print("\nSTEP 3 RESULT")
    for name in ("pca_linear_probe", "plain_mlp", "plain_autoencoder"):
        t = models[name]["test"]
        print(
            f"{name}: AUROC={t['auroc']:.6f}; AUPRC={t['auprc']:.6f}; "
            f"balanced_accuracy={t['balanced_accuracy']:.6f}; "
            f"F1_macro={t['f1_macro']:.6f}; Brier={t['brier_score']:.6f}"
        )
    print("Test biological units:", len(test_samples))
    print("Low-sample warning (<10 test units):", len(test_samples) < 10)
    print("Artifact:", REPORT)
    print("\nSTEP: 3 — Baseline suite")
    print("STATUS: DONE")
    print("ARTIFACT(S): reports/baselines_v1.json")
    print("METRIC(S): " + "; ".join(
        f"{m}.AUROC={models[m]['test']['auroc']:.6f}"
        for m in ("pca_linear_probe", "plain_mlp", "plain_autoencoder")
    ))
    print("ISSUES: none")
    print("NEXT ACTION: awaiting human approval to proceed to Step 4")


if __name__ == "__main__":
    try:
        main()
    except Step3Halt as exc:
        print("\nSTEP: 3 — Baseline suite")
        print("STATUS: BLOCKED")
        print("ARTIFACT(S): reports/baselines_v1.json [not created]")
        print("METRIC(S): none — benchmark did not complete")
        print(f"ISSUES: {exc}")
        print("NEXT ACTION: awaiting human approval to proceed to Step 4")
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("\nSTEP: 3 — Baseline suite")
        print("STATUS: FAILED")
        print("ARTIFACT(S): reports/baselines_v1.json [not created]")
        print("METRIC(S): none — benchmark did not complete")
        print(f"ISSUES: {type(exc).__name__}: {exc}")
        print("NEXT ACTION: awaiting human approval to proceed to Step 4")
        raise
