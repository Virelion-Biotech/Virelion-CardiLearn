"""Training engine for the CardiLearn research representation model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import json
import random

import numpy as np
import torch
from torch import nn

from .objectives import (
    ObjectiveSchedule,
    ObjectiveStage,
    ObjectiveWeights,
    cosine_alignment_loss,
    masked_log1p_loss,
    negative_binomial_nll,
    vicreg_loss,
)
from .research_model import CardiLearnResearch


@dataclass(frozen=True)
class TorchTrainConfig:
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_accumulation_steps: int = 1
    gradient_clip_norm: float = 5.0
    mask_fraction: float = 0.15
    mixed_precision: bool = True
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.grad_accumulation_steps < 1:
            raise ValueError("epochs and grad_accumulation_steps must be positive")
        if not 0 < self.mask_fraction < 1:
            raise ValueError("mask_fraction must be in (0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer hyperparameters")


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    components: dict[str, float]


class CategoryEncoder:
    """Train-only categorical vocabulary with an explicit unknown code."""

    UNK = "<UNK>"

    def __init__(self, values: Iterable[str]) -> None:
        unique = sorted({str(value) for value in values if str(value)})
        self.mapping = {self.UNK: 0}
        self.mapping.update({value: index + 1 for index, value in enumerate(unique)})

    def encode(self, values: Iterable[str]) -> np.ndarray:
        return np.asarray([self.mapping.get(str(value), 0) for value in values], dtype=np.int64)

    def to_dict(self) -> dict[str, int]:
        return dict(self.mapping)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_gene_mask(counts: torch.Tensor, fraction: float) -> torch.Tensor:
    if counts.ndim != 2:
        raise ValueError("counts must be [batch, genes]")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    mask = torch.rand(counts.shape, device=counts.device) < fraction
    empty_rows = ~mask.any(dim=1)
    if empty_rows.any():
        first = torch.zeros(mask.shape[0], dtype=torch.bool, device=counts.device)
        first[empty_rows] = True
        mask[:, 0] |= first
    return mask


def research_batch_loss(
    model: CardiLearnResearch,
    batch: Mapping[str, torch.Tensor],
    weights: ObjectiveWeights,
    *,
    mask_fraction: float = 0.15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    required = {"counts", "species", "assay"}
    missing = required.difference(batch)
    if missing:
        raise ValueError(f"research batch missing keys: {sorted(missing)}")

    counts = batch["counts"].float()
    species = batch["species"].long()
    assay = batch["assay"].long()
    tissue = batch.get("tissue")
    if tissue is not None:
        tissue = tissue.long()
    library_size = batch.get("library_size")
    if library_size is None:
        library_size = counts.sum(dim=-1).clamp_min(1.0)

    mask_a = random_gene_mask(counts, mask_fraction)
    mask_b = random_gene_mask(counts, mask_fraction)
    masked_a = counts.masked_fill(mask_a, 0.0)
    masked_b = counts.masked_fill(mask_b, 0.0)

    out_a = model(
        masked_a,
        species,
        assay,
        tissue=tissue,
        library_size=library_size,
        gene_mask=mask_a,
    )
    out_b = model(
        masked_b,
        species,
        assay,
        tissue=tissue,
        library_size=library_size,
        gene_mask=mask_b,
    )

    zero = counts.new_zeros(())
    l_reconstruction = negative_binomial_nll(
        out_a.reconstruction_mu, out_a.reconstruction_theta, counts
    )
    l_masked = masked_log1p_loss(out_a.masked_prediction, counts, mask_a)
    l_contrastive = cosine_alignment_loss(out_a.z_shared, out_b.z_shared) if weights.contrastive else zero
    l_vicreg = (
        vicreg_loss(out_a.z_shared, out_b.z_shared)[0] if weights.vicreg else zero
    )

    l_cell_type = zero
    l_maturation = zero
    l_injury = zero
    l_species = zero

    if weights.cell_type and "cell_type" in batch:
        target = batch["cell_type"].long()
        valid = target >= 0
        if valid.any():
            l_cell_type = nn.functional.cross_entropy(out_a.cell_type[valid], target[valid])

    if weights.maturation and "maturation" in batch:
        target = batch["maturation"].float()
        valid = torch.isfinite(target)
        if valid.any():
            l_maturation = nn.functional.smooth_l1_loss(out_a.maturation[valid], target[valid])

    if weights.injury and "injury" in batch:
        target = batch["injury"].float()
        valid = torch.isfinite(target)
        if valid.any():
            l_injury = nn.functional.binary_cross_entropy_with_logits(out_a.injury[valid], target[valid])

    if (
        weights.species_adversarial
        and out_a.species_logits is not None
        and species.numel() > 0
    ):
        l_species = nn.functional.cross_entropy(out_a.species_logits, species)

    total = (
        weights.reconstruction * l_reconstruction
        + weights.masked * l_masked
        + weights.contrastive * l_contrastive
        + weights.vicreg * l_vicreg
        + weights.cell_type * l_cell_type
        + weights.maturation * l_maturation
        + weights.injury * l_injury
        + weights.species_adversarial * l_species
    )
    components = {
        "reconstruction": l_reconstruction,
        "masked": l_masked,
        "contrastive": l_contrastive,
        "vicreg": l_vicreg,
        "cell_type": l_cell_type,
        "maturation": l_maturation,
        "injury": l_injury,
        "species_adversarial": l_species,
    }
    return total, components


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    path: str | Path,
    epoch: int,
    config: TorchTrainConfig,
    objective_weights: ObjectiveWeights,
    extra: Mapping[str, object] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "1.0",
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_config": config.__dict__,
            "objective_weights": objective_weights.__dict__,
            "extra": dict(extra or {}),
        },
        destination,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("invalid CardiLearn checkpoint")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        state = payload.get("optimizer_state_dict")
        if state is None:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(state)
    return payload


def fit_research_model(
    model: CardiLearnResearch,
    loader,
    *,
    config: TorchTrainConfig,
    schedule: ObjectiveSchedule | None = None,
    checkpoint_dir: str | Path | None = None,
) -> list[EpochRecord]:
    device = resolve_device(config.device)
    seed_everything(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    use_amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if schedule is None:
        schedule = ObjectiveSchedule(
            (ObjectiveStage("default", ObjectiveWeights()),)
        )

    history: list[EpochRecord] = []
    for epoch in range(1, config.epochs + 1):
        stage = schedule.at_epoch(epoch)
        model.train()
        totals: dict[str, float] = {}
        sample_count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(loader):
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                loss, components = research_batch_loss(
                    model, moved, stage.weights, mask_fraction=config.mask_fraction
                )
                scaled_loss = loss / config.grad_accumulation_steps
            scaler.scale(scaled_loss).backward()

            should_step = (
                (batch_index + 1) % config.grad_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = int(moved["counts"].shape[0])
            sample_count += batch_size
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * batch_size
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size

        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=totals.get("loss", 0.0) / max(sample_count, 1),
                components={
                    name: value / max(sample_count, 1)
                    for name, value in totals.items()
                    if name != "loss"
                },
            )
        )

        if checkpoint_dir is not None:
            save_checkpoint(
                model,
                optimizer,
                path=Path(checkpoint_dir) / f"epoch_{epoch:04d}.pt",
                epoch=epoch,
                config=config,
                objective_weights=stage.weights,
                extra={"history_record": history[-1].__dict__},
            )
    return history


def history_to_json(history: Iterable[EpochRecord], path: str | Path) -> None:
    records = [
        {"epoch": item.epoch, "train_loss": item.train_loss, "components": item.components}
        for item in history
    ]
    Path(path).write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
