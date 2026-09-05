"""Training utilities for CardiLearn prototype and research models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentations import make_two_views
from .losses import (
    anti_collapse_loss,
    cell_type_loss,
    injury_loss,
    invariance_loss,
    maturation_absolute_loss,
    masked_gene_loss,
    negative_binomial_loss,
    reconstruction_loss,
    species_adversarial_loss,
)
from .model import CardiLearnLarge, CardiLearnProto


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    mask_fraction: float = 0.15
    grad_clip: float = 5.0
    recon_weight: float = 1.0
    masked_weight: float = 1.0
    invariance_weight: float = 0.25
    anti_collapse_weight: float = 0.10
    cell_type_weight: float = 0.50
    maturation_weight: float = 1.0
    injury_weight: float = 0.50
    species_adversarial_weight: float = 0.10
    seed: int = 42


def seed_torch(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mask_genes(x: torch.Tensor, fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("mask_fraction must be in [0, 1)")
    mask = torch.rand_like(x) < fraction
    # A masked input is represented by zero counts, which is unambiguous for
    # the count model because zero is a valid observed count.
    return x.masked_fill(mask, 0.0), mask


def train_one_epoch(
    model: CardiLearnProto | CardiLearnLarge,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float]:
    model.train()
    names = ["loss", "recon", "masked", "invariance", "anti_collapse", "cell_type", "maturation", "injury", "species_adversarial"]
    totals = {name: 0.0 for name in names}
    n = 0
    for batch in loader:
        x = torch.as_tensor(batch["x"], dtype=torch.float32, device=device)
        species = torch.as_tensor(batch["species"], dtype=torch.long, device=device)
        assay = torch.as_tensor(batch["assay"], dtype=torch.long, device=device)
        cell_type = torch.as_tensor(batch["cell_type"], dtype=torch.long, device=device)
        maturation = torch.as_tensor(batch["maturation"], dtype=torch.float32, device=device)
        injury = torch.as_tensor(batch["injury"], dtype=torch.float32, device=device)

        x1, x2 = make_two_views(x, config.mask_fraction)
        if isinstance(model, CardiLearnLarge):
            masked_x, gene_mask = _mask_genes(x, config.mask_fraction)
            library_size = torch.clamp(x.sum(dim=-1), min=1.0)
            out1 = model(masked_x, species, assay, library_size=library_size)
            out2 = model(x2, species, assay, library_size=library_size)
            l_recon = negative_binomial_loss(out1.nb_mu, out1.nb_theta, x)
            l_masked = masked_gene_loss(out1.masked_prediction, x, gene_mask)
            l_species = species_adversarial_loss(out1.species_logits, species)
        else:
            out1 = model(x1, species, assay)
            out2 = model(x2, species, assay)
            l_recon = reconstruction_loss(out1.reconstruction, x)
            l_masked = torch.zeros((), device=device)
            l_species = torch.zeros((), device=device)

        l_inv = invariance_loss(out1.z_shared, out2.z_shared)
        l_anti = anti_collapse_loss(out1.z_shared)
        l_cell = cell_type_loss(out1.cell_type, cell_type)
        l_mat = maturation_absolute_loss(out1.maturity, maturation)
        l_injury = injury_loss(out1.injury, injury)
        loss = (
            config.recon_weight * l_recon
            + config.masked_weight * l_masked
            + config.invariance_weight * l_inv
            + config.anti_collapse_weight * l_anti
            + config.cell_type_weight * l_cell
            + config.maturation_weight * l_mat
            + config.injury_weight * l_injury
            + config.species_adversarial_weight * l_species
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        bs = x.shape[0]
        values = {"loss": loss, "recon": l_recon, "masked": l_masked, "invariance": l_inv,
                  "anti_collapse": l_anti, "cell_type": l_cell, "maturation": l_mat,
                  "injury": l_injury, "species_adversarial": l_species}
        for name, value in values.items():
            totals[name] += float(value.detach()) * bs
        n += bs
    return {name: value / max(n, 1) for name, value in totals.items()}


@torch.no_grad()
def encode_dataset(model: CardiLearnProto | CardiLearnLarge, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    z, maturity, injury, cell_type = [], [], [], []
    for batch in loader:
        x = torch.as_tensor(batch["x"], dtype=torch.float32, device=device)
        species = torch.as_tensor(batch["species"], dtype=torch.long, device=device)
        assay = torch.as_tensor(batch["assay"], dtype=torch.long, device=device)
        out = model(x, species, assay)
        z.append(out.z_shared.cpu().numpy())
        maturity.append(out.maturity.cpu().numpy())
        injury.append(out.injury.cpu().numpy())
        cell_type.append(out.cell_type.cpu().numpy())
    return {"z_shared": np.concatenate(z), "maturity": np.concatenate(maturity), "injury": np.concatenate(injury), "cell_type": np.concatenate(cell_type)}
