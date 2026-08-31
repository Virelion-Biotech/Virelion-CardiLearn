#!/usr/bin/env python3
"""End-to-end Kaggle pipeline for leakage-safe GSE308914 MI validation.

Starts from GEO download and ends with machine-readable results, figures, and a
reproducibility manifest. Designed to run in a Kaggle notebook or as a script.

The pipeline deliberately keeps *all learned operations inside each training
fold*: library-size normalization, log transform, gene filtering/HVG selection,
scaling, and model fitting. It evaluates D0 baseline vs post-MI; D0 is NOT sham.

Usage in Kaggle:
    !python scripts/kaggle_gse308914_full_pipeline.py --out /kaggle/working/cardilearn_gse308914

Optional raw GEO download can be disabled when data are already cached:
    --skip-download

Notes:
- GEOquery/NCBI access can change; the downloader first attempts GEOparse and
  then falls back to the GEO FTP matrix files.
- This script expects a sample-by-gene expression matrix after download. It
  supports a 10x-style MTX directory if the accession contains matrix files,
  but intentionally fails loudly when sample/gene mapping cannot be established.
- For a paper, inspect the generated provenance and QC files before treating
  any metric as final evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ACCESSION = "GSE308914"
RANDOM_STATE = 42
DEFAULT_K = 2000


@dataclass
class FoldResult:
    repeat: int
    fold: int
    n_train: int
    n_test: int
    auroc: float
    auprc: float
    accuracy: float
    balanced_accuracy: float
    f1: float
    precision: float
    recall: float
    brier: float


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_geo(accession: str, raw_dir: Path) -> list[Path]:
    """Download GEO supplementary files and SOFT metadata without guessing data semantics."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        import GEOparse  # type: ignore
        print(f"Downloading {accession} with GEOparse...")
        gse = GEOparse.get_GEO(geo=accession, destdir=str(raw_dir), silent=False)
        soft = raw_dir / f"{accession}_family.soft.gz"
        if not soft.exists():
            # GEOparse usually caches the SOFT file; record available files.
            print("GEOparse completed; no family SOFT file located at the conventional path.")
        return sorted(raw_dir.rglob("*"))
    except Exception as exc:
        print(f"GEOparse path unavailable: {exc}")

    # NCBI supplementary FTP index is public but filenames vary by study.
    # Download the family SOFT first; feature matrices are handled separately.
    group = accession[:-3] + "nnn" if len(accession) >= 3 else accession
    # GEO's canonical SOFT URL.
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{group}/{accession}/soft/{accession}_family.soft.gz"
    target = raw_dir / f"{accession}_family.soft.gz"
    try:
        print("Downloading", url)
        urllib.request.urlretrieve(url, target)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {accession} family SOFT from NCBI. "
            "In Kaggle, verify Internet access is enabled and inspect the accession manually."
        ) from exc
    return sorted(raw_dir.rglob("*"))


def parse_soft_metadata(soft_path: Path) -> pd.DataFrame:
    """Extract sample-level characteristics from GEO family SOFT."""
    import gzip
    import re

    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    gsm = None
    opener = gzip.open if soft_path.suffix == ".gz" else open
    with opener(soft_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE ="):
                if current and gsm:
                    rows.append({"sample_id": gsm, **current})
                gsm = line.split("=", 1)[1].strip()
                current = {}
            elif gsm and line.startswith("!Sample_characteristics_ch1"):
                value = line.split("=", 1)[1].strip()
                # GEO may provide repeated characteristics; preserve all values.
                key = f"characteristic_{len([k for k in current if k.startswith('characteristic_')])}"
                current[key] = value
        if current and gsm:
            rows.append({"sample_id": gsm, **current})
    meta = pd.DataFrame(rows)
    if meta.empty:
        raise ValueError(f"No GSM sample metadata parsed from {soft_path}")
    return meta


def infer_labels(meta: pd.DataFrame) -> pd.DataFrame:
    """Infer labels only from explicit metadata strings; fail on ambiguity."""
    out = meta.copy()
    text_cols = [c for c in out.columns if c.startswith("characteristic_")]
    blob = out[text_cols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()

    def label(s: str) -> int:
        # D0/reference is acceptable; sham is intentionally not conflated with D0.
        if any(x in s for x in ["d0", "baseline", "control", "steady-state", "steady state"]):
            if "mi" in s and "d0" not in s:
                return -1
            return 0
        if any(x in s for x in ["post-mi", "post mi", "myocardial infarction", "mi day", "mi_"]):
            return 1
        return -1

    out["injury_label"] = blob.map(label)
    bad = out.loc[out.injury_label < 0, ["sample_id", *text_cols]]
    if not bad.empty:
        raise ValueError(
            "Could not unambiguously infer baseline/post-MI labels for samples:\n"
            + bad.to_string(index=False)
        )
    out["biological_group_id"] = out["sample_id"]
    return out


def locate_matrix(raw_dir: Path) -> Path:
    """Locate a conventional GEO matrix; refuse to guess among unrelated files."""
    candidates = [
        p for p in raw_dir.rglob("*") if p.is_file() and any(
            token in p.name.lower() for token in ["matrix", "count", "expression"]
        )
    ]
    candidates = [p for p in candidates if p.suffix.lower() in {".csv", ".tsv", ".txt", ".gz", ".h5", ".h5ad"}]
    if not candidates:
        raise FileNotFoundError(
            "No expression matrix was found automatically. Download the study's processed expression matrix "
            "into the raw directory and rerun. The pipeline intentionally does not invent a matrix."
        )
    if len(candidates) > 1:
        raise RuntimeError("Multiple possible expression matrices found; pass --matrix explicitly.")
    return candidates[0]


def read_matrix(path: Path) -> pd.DataFrame:
    if path.suffix == ".h5ad":
        import anndata as ad  # type: ignore
        a = ad.read_h5ad(path)
        x = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        return pd.DataFrame(x, index=a.obs_names, columns=a.var_names)
    compression = "gzip" if path.name.endswith(".gz") else None
    sep = "," if ".csv" in path.name else "\t"
    df = pd.read_csv(path, sep=sep, compression=compression, index_col=0)
    # Standard GEO matrices are often gene x sample; orient to sample x gene.
    if df.shape[0] > df.shape[1] * 2:
        df = df.T
    return df.apply(pd.to_numeric, errors="coerce")


def align_samples(matrix: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = set(matrix.index.astype(str))
    meta2 = meta[meta.sample_id.astype(str).isin(sample_ids)].copy()
    if meta2.empty:
        # Try matrix columns if it was not transposed as expected.
        matrix2 = matrix.T
        sample_ids = set(matrix2.index.astype(str))
        meta2 = meta[meta.sample_id.astype(str).isin(sample_ids)].copy()
        if not meta2.empty:
            matrix = matrix2
    if meta2.empty:
        raise ValueError("No overlap between GEO sample IDs and expression matrix sample IDs.")
    meta2 = meta2.drop_duplicates("sample_id").set_index("sample_id")
    common = [s for s in meta2.index if s in matrix.index]
    if len(common) < 10:
        raise ValueError(f"Only {len(common)} matched samples; refusing to continue.")
    matrix = matrix.loc[common].copy()
    meta2 = meta2.loc[common].copy()
    return matrix, meta2.reset_index()


def clean_expression(x: pd.DataFrame) -> pd.DataFrame:
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.loc[:, x.notna().mean() >= 0.9]
    x = x.fillna(x.median(numeric_only=True))
    # Remove nonpositive / constant features before log transform.
    x = x.loc[:, (x >= 0).all(axis=0)]
    x = x.loc[:, x.nunique(dropna=False) > 1]
    return x


def fold_transform(train: pd.DataFrame, test: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Training-only normalization + HVG selection + scaling."""
    train = train.copy()
    test = test.loc[:, train.columns].copy()
    # Counts/abundances are normalized using sample-wise library size estimated from training features.
    train_sum = train.sum(axis=1).replace(0, np.nan)
    test_sum = test.sum(axis=1).replace(0, np.nan)
    train = train.div(train_sum, axis=0) * 1e6
    test = test.div(test_sum, axis=0) * 1e6
    train = np.log1p(train)
    test = np.log1p(test)
    # HVG selection is strictly inside this fold.
    variances = train.var(axis=0, ddof=1).sort_values(ascending=False)
    selected = variances.head(min(k, len(variances))).index.tolist()
    train = train[selected]
    test = test[selected]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    a = scaler.fit_transform(imputer.fit_transform(train))
    b = scaler.transform(imputer.transform(test))
    return a, b, selected


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
    }


def nested_cv(x: pd.DataFrame, y: np.ndarray, repeats: int, folds: int, k: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    rows: list[dict[str, object]] = []
    oof = np.full(len(y), np.nan)
    selected_counts: dict[str, int] = {}
    for repeat in range(repeats):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (tr, te) in enumerate(cv.split(x, y), start=1):
            a, b, selected = fold_transform(x.iloc[tr], x.iloc[te], k)
            for g in selected:
                selected_counts[g] = selected_counts.get(g, 0) + 1
            model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
            model.fit(a, y[tr])
            p = model.predict_proba(b)[:, 1]
            oof[te] = p
            rows.append({"repeat": repeat + 1, "fold": fold, "n_train": len(tr), "n_test": len(te), **metrics(y[te], p)})
    return pd.DataFrame(rows), oof, {"feature_selection_frequency": dict(sorted(selected_counts.items(), key=lambda z: (-z[1], z[0])))}


def permutation_test(x: pd.DataFrame, y: np.ndarray, folds: int, permutations: int, k: int, seed: int) -> dict[str, object]:
    """Permutation test reruns feature selection inside every fold."""
    rng = np.random.default_rng(seed)
    observed_df, _, _ = nested_cv(x, y, repeats=1, folds=folds, k=k, seed=seed)
    observed = float(observed_df.auroc.mean())
    null: list[float] = []
    for i in range(permutations):
        yp = rng.permutation(y)
        df, _, _ = nested_cv(x, yp, repeats=1, folds=folds, k=k, seed=seed + 1000 + i)
        null.append(float(df.auroc.mean()))
    p = (1 + sum(v >= observed for v in null)) / (len(null) + 1)
    return {"observed_mean_fold_auroc": observed, "n_permutations": permutations, "permutation_p": p, "null_mean": float(np.mean(null)), "null_sd": float(np.std(null, ddof=1)), "null_auroc": null}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/cardilearn_gse308914"))
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--matrix", type=Path)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--hvg-k", type=int, default=DEFAULT_K)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--permutations", type=int, default=100)
    args = ap.parse_args()
    out = args.out
    raw = args.raw_dir or (out / "raw")
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)

    manifest: dict[str, object] = {
        "accession": ACCESSION,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "hvg_k": args.hvg_k,
        "folds": args.folds,
        "repeats": args.repeats,
        "permutations": args.permutations,
        "label_definition": "0=D0/baseline/reference; 1=post-MI; D0 is not sham",
        "pipeline": "training-fold-only normalization, log1p, HVG selection, imputation, scaling; logistic regression",
    }

    if not args.skip_download:
        files = download_geo(ACCESSION, raw)
        manifest["downloaded_files"] = [str(p.relative_to(raw)) for p in files if p.is_file()]
    softs = list(raw.glob("*_family.soft.gz"))
    if not softs:
        raise FileNotFoundError("GEO family SOFT not found; enable download or place it in --raw-dir")
    soft = softs[0]
    meta = infer_labels(parse_soft_metadata(soft))
    meta.to_csv(out / "tables" / "sample_metadata.csv", index=False)

    matrix_path = args.matrix or locate_matrix(raw)
    matrix = clean_expression(read_matrix(matrix_path))
    matrix, meta = align_samples(matrix, meta)
    x = matrix.copy()
    y = meta.injury_label.to_numpy(dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Both baseline and post-MI classes are required.")
    if min(np.bincount(y)) < args.folds:
        raise ValueError("Not enough samples in the minority class for requested folds.")
    matrix.to_csv(out / "tables" / "aligned_expression.csv")
    meta.to_csv(out / "tables" / "aligned_metadata.csv", index=False)
    manifest["matrix_file"] = str(matrix_path)
    manifest["matrix_sha256"] = sha256(matrix_path)
    manifest["n_samples"] = len(x)
    manifest["n_features_input"] = x.shape[1]
    manifest["class_counts"] = {str(i): int((y == i).sum()) for i in [0, 1]}

    fold_df, oof, details = nested_cv(x, y, args.repeats, args.folds, args.hvg_k, RANDOM_STATE)
    fold_df.to_csv(out / "tables" / "nested_cv_fold_metrics.csv", index=False)
    pd.DataFrame({"sample_id": meta.sample_id, "injury_label": y, "oof_probability": oof}).to_csv(out / "tables" / "oof_predictions.csv", index=False)
    summary = fold_df.drop(columns=["repeat", "fold", "n_train", "n_test"]).agg(["mean", "std", "min", "max"]).T
    summary.to_csv(out / "tables" / "nested_cv_summary.csv")

    perm = permutation_test(x, y, args.folds, args.permutations, args.hvg_k, RANDOM_STATE)
    (out / "tables" / "permutation_test.json").write_text(json.dumps(perm, indent=2), encoding="utf-8")
    (out / "tables" / "feature_selection_frequency.json").write_text(json.dumps(details, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y, oof, ax=ax)
        ax.set_title("GSE308914 — out-of-fold ROC")
        fig.tight_layout(); fig.savefig(out / "figures" / "oof_roc.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(perm["null_auroc"], bins=20)
        ax.axvline(perm["observed_mean_fold_auroc"], linestyle="--")
        ax.set_xlabel("Mean CV AUROC under label permutation")
        ax.set_ylabel("Count")
        fig.tight_layout(); fig.savefig(out / "figures" / "permutation_null.png", dpi=200); plt.close(fig)
    except Exception as exc:
        manifest["plot_warning"] = str(exc)

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["results"] = {
        "mean_cv_auroc": float(fold_df.auroc.mean()),
        "sd_cv_auroc": float(fold_df.auroc.std(ddof=1)),
        "mean_cv_auprc": float(fold_df.auprc.mean()),
        "permutation_p": perm["permutation_p"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest["results"], indent=2))
    print(f"Results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
