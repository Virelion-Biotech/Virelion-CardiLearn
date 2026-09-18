"""Train CardiLearn on an explicitly curated real-data matrix.

large uses the current modular research architecture; proto keeps the small legacy prototype.
No raw expression or generated model artifact is committed by this script.
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
        with np.load(path, allow_pickle=False) as payload:
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
    if not np.isfinite(X).all() or np.any(X < 0):
        raise ValueError("expression matrix must be finite and non-negative")
    if len(set(genes)) != len(genes):
        raise ValueError("expression gene names must be unique")
    return X, genes


def _require_metadata(metadata: pd.DataFrame) -> None:
    required = {"study_id", "subject_id", "sample_id", "species", "assay", "cell_type", "maturation", "injury"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"training metadata missing columns: {sorted(missing)}")
    if metadata[list(required)].isna().any().any():
        raise ValueError("required training metadata cannot contain missing values")
    for column in ("study_id", "subject_id", "sample_id", "species", "assay", "cell_type"):
        if metadata[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} contains empty identifiers/labels")
    for child, parent in (("sample_id", "subject_id"), ("subject_id", "study_id")):
        mapping_counts = metadata.groupby(child, dropna=False)[parent].nunique(dropna=False)
        if (mapping_counts > 1).any():
            bad = mapping_counts[mapping_counts > 1].index.tolist()[:10]
            raise ValueError(f"{child} maps to multiple {parent} values: {bad}")


def _split_metadata(metadata: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    group_column = "study_family_id" if "study_family_id" in metadata.columns else "study_id"
    splits = study_split(metadata, seed=seed, group_column=group_column)
    assigned = assign_split(metadata, splits, group_column=group_column)
    assert_no_hierarchy_leakage(assigned)
    return assigned, splits


def _binary_train_mapping(values: pd.Series) -> dict[str, int]:
    labels = sorted(values.astype(str).unique().tolist())
    if len(labels) != 2:
        raise ValueError(f"injury head is binary, but training contains {len(labels)} classes: {labels}")
    return {label: index for index, label in enumerate(labels)}


def _encode_research_metadata(metadata: pd.DataFrame, train_mask: np.ndarray):
    from cardilearn.torch_training import CategoryEncoder

    train = metadata.loc[train_mask]
    species_encoder = CategoryEncoder(train["species"])
    assay_encoder = CategoryEncoder(train["assay"])
    cell_type_encoder = CategoryEncoder(train["cell_type"])
    injury_mapping = _binary_train_mapping(train["injury"])
    encoded = metadata.copy()
    encoded["species"] = species_encoder.encode(encoded["species"])
    encoded["assay"] = assay_encoder.encode(encoded["assay"])
    encoded["cell_type"] = cell_type_encoder.encode(encoded["cell_type"])
    encoded["injury"] = encoded["injury"].astype(str).map(injury_mapping).fillna(-1).astype(np.int64)
    encoded["maturation"] = pd.to_numeric(encoded["maturation"], errors="raise").astype(np.float32)
    if encoded.loc[train_mask, ["species", "assay", "cell_type", "injury"]].lt(0).any().any():
        raise ValueError("training partition contains an invalid encoded category")
    return encoded, species_encoder.to_dict(), assay_encoder.to_dict(), cell_type_encoder.to_dict(), injury_mapping


def _research_config(model) -> dict[str, object]:
    return {
        "class": model.__class__.__name__,
        "n_genes": model.n_genes,
        "gene_dim": model.gene_dim,
        "n_programs": model.n_programs,
        "shared_dim": model.shared_dim,
        "private_dim": model.private_dim,
        "decoder_dim": model.decoder.decoder_dim,
        "backbone": model.program_backbone.backend,
        "species_adversarial_strength": model.species_adversarial_strength,
        "parameter_count": model.parameter_count(),
    }


def _run_research(X: np.ndarray, metadata: pd.DataFrame, train_mask: np.ndarray, *, args, out: Path) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader
    from cardilearn.objectives import ObjectiveSchedule, ObjectiveStage, ObjectiveWeights
    from cardilearn.research_model import CardiLearnResearch
    from cardilearn.torch_training import MappingTensorDataset, TorchTrainConfig, fit_research_model, history_to_json

    encoded, species_codes, assay_codes, cell_codes, injury_codes = _encode_research_metadata(metadata, train_mask)
    train_meta = encoded.loc[train_mask].reset_index(drop=True)
    dataset = MappingTensorDataset({
        "counts": torch.as_tensor(X[train_mask], dtype=torch.float32),
        "species": torch.as_tensor(train_meta["species"].to_numpy(), dtype=torch.long),
        "assay": torch.as_tensor(train_meta["assay"].to_numpy(), dtype=torch.long),
        "cell_type": torch.as_tensor(train_meta["cell_type"].to_numpy(), dtype=torch.long),
        "maturation": torch.as_tensor(train_meta["maturation"].to_numpy(), dtype=torch.float32),
        "injury": torch.as_tensor(train_meta["injury"].to_numpy(), dtype=torch.float32),
    })
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = CardiLearnResearch(
        n_genes=X.shape[1],
        n_species=len(species_codes),
        n_assays=len(assay_codes),
        n_cell_types=len(cell_codes),
        gene_dim=256, n_programs=128, n_layers=6, n_heads=8,
        shared_dim=384, private_dim=128, decoder_dim=256,
    )
    config = TorchTrainConfig(epochs=args.epochs, seed=args.seed, device="auto", mixed_precision=True, mask_fraction=0.15)
    schedule = ObjectiveSchedule((
        ObjectiveStage("representation", ObjectiveWeights(reconstruction=1.0, masked=1.0, contrastive=0.0, vicreg=0.10, cell_type=0.0, maturation=0.0, injury=0.0, species_adversarial=0.0)),
        ObjectiveStage("biological_state", ObjectiveWeights(reconstruction=1.0, masked=1.0, contrastive=0.25, vicreg=0.10, cell_type=0.50, maturation=1.0, injury=0.50, species_adversarial=0.0)),
    ))
    history = fit_research_model(model, loader, config=config, schedule=schedule, checkpoint_dir=out / "checkpoints")
    device = str(next(model.parameters()).device)
    torch.save({
        "schema_version": "1.0",
        "model_state_dict": model.state_dict(),
        "model_config": _research_config(model),
        "category_codes": {"species": species_codes, "assay": assay_codes, "cell_type": cell_codes, "injury": injury_codes},
    }, out / "model.pt")
    history_to_json(history, out / "training_history.json")
    encoded.to_parquet(out / "split_metadata.parquet", index=False)
    return {"architecture": "CardiLearnResearch", "device": device, "model_config": _research_config(model), "species_codes": species_codes, "assay_codes": assay_codes, "cell_type_codes": cell_codes, "injury_codes": injury_codes}


def _run_proto(X: np.ndarray, metadata: pd.DataFrame, train_mask: np.ndarray, *, args, out: Path) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader
    from cardilearn.prototype.model import CardiLearnProto
    from cardilearn.prototype.train import TrainConfig, seed_torch, train_one_epoch
    encoded, species_codes, assay_codes, cell_codes, _ = _encode_research_metadata(metadata, train_mask)
    dataset = CardiLearnCellDataset(X[train_mask], encoded.loc[train_mask].reset_index(drop=True))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CardiLearnProto(n_genes=X.shape[1], n_species=len(species_codes), n_assays=len(assay_codes), n_cell_types=len(cell_codes)).to(device)
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    for epoch in range(args.epochs):
        metrics = train_one_epoch(model, loader, optimizer, device, config)
        metrics["epoch"] = float(epoch + 1)
        history.append(metrics)
    torch.save(model.state_dict(), out / "model.pt")
    encoded.to_parquet(out / "split_metadata.parquet", index=False)
    (out / "training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"architecture": "CardiLearnProto", "device": str(device), "species_codes": species_codes, "assay_codes": assay_codes, "cell_type_codes": cell_codes}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CardiLearn on locked real-data input")
    parser.add_argument("--expression", required=True, help="training-ready expression matrix")
    parser.add_argument("--metadata", required=True, help="training-ready metadata CSV/Parquet")
    parser.add_argument("--output", default="runs/real-data-v1")
    parser.add_argument("--n-genes", type=int, default=20000)
    parser.add_argument("--model-size", choices=("large", "proto"), default="large")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit("PyTorch is required: install the [torch] extra") from exc
    X_raw, genes = _read_expression(Path(args.expression))
    metadata_path = Path(args.metadata)
    metadata = pd.read_parquet(metadata_path) if metadata_path.suffix == ".parquet" else pd.read_csv(metadata_path)
    _require_metadata(metadata)
    if len(metadata) != X_raw.shape[0]:
        raise ValueError("expression rows must match metadata rows")
    metadata, splits = _split_metadata(metadata, args.seed)
    selected = select_genes_train_only(X_raw, metadata, min(args.n_genes, X_raw.shape[1]))
    X = X_raw[:, selected]
    selected_genes = [genes[index] for index in selected]
    train_mask = metadata["_split"].eq("train").to_numpy()
    if train_mask.sum() < 2:
        raise ValueError("training split contains fewer than two observations")
    model_outputs = _run_research(X, metadata, train_mask, args=args, out=out) if args.model_size == "large" else _run_proto(X, metadata, train_mask, args=args, out=out)
    manifest = {
        "status": "trained_real_data", "model_size": args.model_size, **model_outputs,
        "expression_fingerprint": dataframe_fingerprint(pd.DataFrame(X_raw)),
        "selected_expression_fingerprint": dataframe_fingerprint(pd.DataFrame(X)),
        "metadata_fingerprint": dataframe_fingerprint(metadata),
        "split_fingerprint": fingerprint_ids([f"{key}:{value}" for key, values in splits.items() if key != "group_column" for value in values]),
        "selected_genes": selected_genes, "splits": splits, "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "n_observations": int(X.shape[0]), "n_training_observations": int(train_mask.sum()), "n_genes": int(X.shape[1]),
        "scientific_note": "Training completion is not evidence of biological validity; locked held-out evaluation is required.",
    }
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "model_size": args.model_size, "n_observations": int(X.shape[0]), "n_training_observations": int(train_mask.sum()), "n_genes": int(X.shape[1])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())