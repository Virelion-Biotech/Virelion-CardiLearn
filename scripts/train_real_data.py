"""Train CardiLearn on an explicitly curated real-data matrix.

The runner refuses unresolved metadata, missing biological targets, non-finite
expression, or insufficient independent studies. Splitting and feature
selection happen before fitting, and only the training partition is passed to
the optimizer. Raw expression data are never written to Git.

The default research configuration targets CardiLearnLarge. The smaller
CardiLearnProto remains available with ``--model-size proto`` for software
smoke tests and CPU development.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cardilearn.prototype.data import CardiLearnCellDataset, select_genes_train_only
from cardilearn.prototype.splits import assign_split, assert_no_hierarchy_leakage, study_split
from cardilearn.reproducibility import dataframe_fingerprint, fingerprint_ids


def _read_expression(path: Path) -> tuple[np.ndarray, list[str]]:
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        if "X" not in payload or "genes" not in payload:
            raise ValueError("NPZ must contain X and genes arrays")
        X = np.asarray(payload["X"], dtype=np.float32)
        genes = [str(x) for x in payload["genes"].tolist()]
    elif path.suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
        genes = [str(c) for c in frame.columns]
        X = frame.to_numpy(dtype=np.float32)
    else:
        raise ValueError("expression must be .npz, .csv, or .tsv")
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("expression matrix must be non-empty and two-dimensional")
    if not np.isfinite(X).all():
        raise ValueError("expression matrix contains non-finite values")
    if len(set(genes)) != len(genes):
        raise ValueError("expression gene names must be unique")
    return X, genes


def _require_metadata(metadata: pd.DataFrame) -> None:
    required = {
        "study_id", "subject_id", "sample_id", "species", "assay",
        "cell_type", "maturation", "injury",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"training metadata missing columns: {sorted(missing)}")
    if metadata[["study_id", "subject_id", "sample_id"]].isna().any().any():
        raise ValueError("study_id, subject_id, and sample_id cannot be missing")

    for child, parent in (("sample_id", "subject_id"), ("subject_id", "study_id")):
        mapping_counts = metadata.groupby(child)[parent].nunique(dropna=False)
        if (mapping_counts > 1).any():
            bad = mapping_counts[mapping_counts > 1].index.tolist()[:10]
            raise ValueError(f"{child} maps to multiple {parent} values: {bad}")

    if metadata[["maturation", "injury"]].isna().any().any():
        raise ValueError("maturation/injury labels must be resolved before training")
    if metadata["cell_type"].isna().any():
        raise ValueError("cell_type labels must be resolved before training")
    for column in ("species", "assay", "cell_type"):
        if metadata[column].isna().any():
            raise ValueError(f"{column} labels must be resolved before training")


def _train_only_codes(metadata: pd.DataFrame, train_mask: np.ndarray) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Build categorical vocabularies from training rows only."""
    train = metadata.loc[train_mask]
    mappings = []
    for column in ("species", "assay", "cell_type"):
        values = sorted(train[column].astype(str).unique())
        mappings.append({value: i for i, value in enumerate(values)})
    return tuple(mappings)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CardiLearn on locked real-data input")
    parser.add_argument("--expression", required=True, help="training-ready expression matrix")
    parser.add_argument("--metadata", required=True, help="training-ready sample/cell metadata")
    parser.add_argument("--output", default="runs/real-data-v1")
    parser.add_argument("--n-genes", type=int, default=20000)
    parser.add_argument("--model-size", choices=("large", "proto"), default="large")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("PyTorch is required: install the [torch] extra") from exc

    X_raw, genes = _read_expression(Path(args.expression))
    metadata = pd.read_parquet(args.metadata) if str(args.metadata).endswith(".parquet") else pd.read_csv(args.metadata)
    _require_metadata(metadata)
    if len(metadata) != X_raw.shape[0]:
        raise ValueError("expression rows must match metadata rows")

    splits = study_split(metadata, seed=args.seed, group_column="study_id")
    metadata = assign_split(metadata, splits, group_column="study_id")
    assert_no_hierarchy_leakage(metadata)
    selected = select_genes_train_only(X_raw, metadata, min(args.n_genes, X_raw.shape[1]))
    X = X_raw[:, selected]
    genes = [genes[i] for i in selected]

    train_mask = metadata["_split"].eq("train").to_numpy()
    if train_mask.sum() < 2:
        raise ValueError("training split contains fewer than two observations")
    species_codes, assay_codes, cell_codes = _train_only_codes(metadata, train_mask)

    encoded = metadata.copy()
    for column, mapping in (("species", species_codes), ("assay", assay_codes), ("cell_type", cell_codes)):
        encoded[column] = encoded[column].astype(str).map(mapping)
    train_metadata = encoded.loc[train_mask].reset_index(drop=True)
    train_X = X[train_mask]
    if train_metadata[["species", "assay", "cell_type"]].isna().any().any():
        raise ValueError("training rows contain categories that could not be encoded")

    dataset = CardiLearnCellDataset(train_X, train_metadata)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    from cardilearn.prototype.model import CardiLearnLarge, CardiLearnProto
    from cardilearn.prototype.train import TrainConfig, seed_torch, train_one_epoch

    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_size == "large":
        model = CardiLearnLarge(
            n_genes=X.shape[1],
            n_species=len(species_codes),
            n_assays=len(assay_codes),
            n_cell_types=len(cell_codes),
        ).to(device)
    else:
        model = CardiLearnProto(
            n_genes=X.shape[1],
            n_species=len(species_codes),
            n_assays=len(assay_codes),
            n_cell_types=len(cell_codes),
        ).to(device)

    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        metrics = train_one_epoch(model, loader, optimizer, device, config)
        metrics["epoch"] = float(epoch + 1)
        history.append(metrics)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    encoded.to_parquet(out / "split_metadata.parquet", index=False)
    (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    manifest = {
        "status": "trained_real_data",
        "model_size": args.model_size,
        "architecture": "CardiLearnLarge" if args.model_size == "large" else "CardiLearnProto",
        "expression_fingerprint": dataframe_fingerprint(pd.DataFrame(X_raw)),
        "selected_expression_fingerprint": dataframe_fingerprint(pd.DataFrame(X)),
        "metadata_fingerprint": dataframe_fingerprint(encoded),
        "split_fingerprint": fingerprint_ids([f"{k}:{v}" for k, values in splits.items() if k != "group_column" for v in values]),
        "selected_genes": genes,
        "species_codes": species_codes,
        "assay_codes": assay_codes,
        "cell_type_codes": cell_codes,
        "splits": splits,
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "n_observations": int(X.shape[0]),
        "n_training_observations": int(train_mask.sum()),
        "n_genes": int(X.shape[1]),
        "scientific_note": "Training completion is not evidence of biological validity; locked held-out evaluation is required.",
    }
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(out), "model_size": args.model_size, "device": str(device), "n_observations": len(X), "n_training_observations": int(train_mask.sum()), "n_genes": X.shape[1]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
