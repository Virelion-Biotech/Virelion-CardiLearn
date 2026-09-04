"""Train the CardiLearn prototype on an explicitly curated real-data matrix.

The runner refuses unresolved metadata, missing biological targets, non-finite
expression, or insufficient independent studies. Splitting and feature
selection happen before fitting, and only the training partition is passed to
the optimizer. Raw expression data are never written to Git.
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
    if metadata["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique")
    if metadata[["maturation", "injury"]].isna().any().any():
        raise ValueError("maturation/injury labels must be resolved before training")
    if metadata["cell_type"].isna().any():
        raise ValueError("cell_type labels must be resolved before training")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CardiLearn on locked real-data input")
    parser.add_argument("--expression", required=True, help="training-ready expression matrix")
    parser.add_argument("--metadata", required=True, help="training-ready sample/cell metadata")
    parser.add_argument("--output", default="runs/real-data-v1")
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("PyTorch is required: install the [torch] extra") from exc

    X, genes = _read_expression(Path(args.expression))
    metadata = pd.read_parquet(args.metadata) if str(args.metadata).endswith(".parquet") else pd.read_csv(args.metadata)
    _require_metadata(metadata)
    if len(metadata) != X.shape[0]:
        raise ValueError("expression rows must match metadata rows")

    splits = study_split(metadata, seed=args.seed, group_column="study_id")
    metadata = assign_split(metadata, splits, group_column="study_id")
    assert_no_hierarchy_leakage(metadata)
    selected = select_genes_train_only(X, metadata, min(args.n_genes, X.shape[1]))
    X = X[:, selected]
    genes = [genes[i] for i in selected]

    species_codes = {v: i for i, v in enumerate(sorted(metadata["species"].astype(str).unique()))}
    assay_codes = {v: i for i, v in enumerate(sorted(metadata["assay"].astype(str).unique()))}
    cell_codes = {v: i for i, v in enumerate(sorted(metadata["cell_type"].astype(str).unique()))}
    encoded = metadata.copy()
    encoded["species"] = encoded["species"].astype(str).map(species_codes)
    encoded["assay"] = encoded["assay"].astype(str).map(assay_codes)
    encoded["cell_type"] = encoded["cell_type"].astype(str).map(cell_codes)

    train_mask = encoded["_split"].eq("train").to_numpy()
    if train_mask.sum() < 2:
        raise ValueError("training split contains fewer than two observations")

    # Critical leakage boundary: validation/test rows are retained only for
    # provenance. They never enter the optimizer or train-time preprocessing.
    train_metadata = encoded.loc[train_mask].reset_index(drop=True)
    train_X = X[train_mask]
    dataset = CardiLearnCellDataset(train_X, train_metadata)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    from cardilearn.prototype.model import CardiLearnProto
    from cardilearn.prototype.train import TrainConfig, seed_torch, train_one_epoch

    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        "expression_fingerprint": dataframe_fingerprint(pd.DataFrame(X)),
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
    print(json.dumps({"output": str(out), "device": str(device), "n_observations": len(X), "n_training_observations": int(train_mask.sum()), "n_genes": X.shape[1]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
