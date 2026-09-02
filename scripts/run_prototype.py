"""Run the CardiLearn v0.1 synthetic smoke experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score
from torch.utils.data import DataLoader

from cardilearn.prototype.data import CardiLearnCellDataset
from cardilearn.prototype.model import CardiLearnProto
from cardilearn.prototype.splits import assert_no_hierarchy_leakage, assign_split, study_split
from cardilearn.prototype.synthetic import SyntheticConfig, generate_synthetic_cardiac_data
from cardilearn.prototype.train import TrainConfig, encode_dataset, seed_torch, train_one_epoch


def evaluate_linear_probes(z_train, z_test, train_meta, test_meta) -> dict[str, float]:
    maturity = Ridge(alpha=1.0).fit(z_train, train_meta["maturation"])
    cell = LogisticRegression(max_iter=2000).fit(z_train, train_meta["cell_type"])
    injury = LogisticRegression(max_iter=2000).fit(z_train, train_meta["injury"])
    maturity_pred = maturity.predict(z_test)
    cell_pred = cell.predict(z_test)
    injury_pred = injury.predict_proba(z_test)[:, 1]
    return {
        "maturation_mae": float(mean_absolute_error(test_meta["maturation"], maturity_pred)),
        "cell_type_balanced_accuracy": float(balanced_accuracy_score(test_meta["cell_type"], cell_pred)),
        "injury_auroc": float(roc_auc_score(test_meta["injury"], injury_pred)),
    }


def main() -> None:
    seed_torch(42)
    X, obs = generate_synthetic_cardiac_data(SyntheticConfig())
    splits = study_split(obs, seed=42)
    meta = assign_split(obs, splits)
    assert_no_hierarchy_leakage(meta)

    train_mask = meta["_split"].eq("train").to_numpy()
    val_mask = meta["_split"].eq("validation").to_numpy()
    test_mask = meta["_split"].eq("test").to_numpy()

    # PCA baseline.
    pca = PCA(n_components=128, random_state=42)
    z_train_pca = pca.fit_transform(X[train_mask])
    z_test_pca = pca.transform(X[test_mask])
    pca_metrics = evaluate_linear_probes(
        z_train_pca,
        z_test_pca,
        meta.loc[train_mask].reset_index(drop=True),
        meta.loc[test_mask].reset_index(drop=True),
    )

    train_ds = CardiLearnCellDataset(X[train_mask], meta.loc[train_mask])
    val_ds = CardiLearnCellDataset(X[val_mask], meta.loc[val_mask])
    test_ds = CardiLearnCellDataset(X[test_mask], meta.loc[test_mask])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = CardiLearnProto(
        n_genes=X.shape[1],
        n_species=int(obs["species"].max()) + 1,
        n_assays=int(obs["assay"].max()) + 1,
        n_cell_types=int(obs["cell_type"].nunique()),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    config = TrainConfig(epochs=10)
    history = []
    for epoch in range(config.epochs):
        metrics = train_one_epoch(model, train_loader, optimizer, torch.device("cpu"), config)
        history.append(metrics)
        print(f"epoch={epoch + 1:02d} loss={metrics['loss']:.5f} maturity={metrics['maturation']:.5f}")

    encoded_train = encode_dataset(model, train_loader, torch.device("cpu"))
    encoded_test = encode_dataset(model, test_loader, torch.device("cpu"))
    cardi_metrics = evaluate_linear_probes(
        encoded_train["z_shared"],
        encoded_test["z_shared"],
        meta.loc[train_mask].reset_index(drop=True),
        meta.loc[test_mask].reset_index(drop=True),
    )

    result = {
        "synthetic": {
            "cells": int(X.shape[0]),
            "genes": int(X.shape[1]),
            "studies": int(obs["study_id"].nunique()),
            "splits": splits,
        },
        "pca_baseline": pca_metrics,
        "cardilearn": cardi_metrics,
        "training": history,
    }
    path = Path("runs/prototype-v0.1")
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), path / "model.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
