"""Typed contracts for perturbation-response experiments."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PerturbationSpec:
    """Experimental perturbation descriptor.

    ``perturbation_id`` is an experiment-defined categorical identifier (drug,
    genetic perturbation, cytokine, environmental stressor, etc.). Dose and
    duration are optional continuous descriptors in dataset-native units.
    """

    perturbation_id: int
    perturbation_type: int = 0
    dose: float = 0.0
    duration: float = 0.0


@dataclass
class PerturbationBatch:
    """Batch tensors consumed by :class:`PerturbationPredictor`."""

    baseline_z: torch.Tensor
    perturbation_id: torch.Tensor
    perturbation_type: torch.Tensor
    dose: torch.Tensor
    duration: torch.Tensor
    target_delta_z: torch.Tensor | None = None

    def __post_init__(self) -> None:
        batch = self.baseline_z.shape[0]
        for name in ("perturbation_id", "perturbation_type", "dose", "duration"):
            value = getattr(self, name)
            if value.ndim != 1 or value.shape[0] != batch:
                raise ValueError(f"{name} must have shape [batch]")
        if self.target_delta_z is not None and self.target_delta_z.shape != self.baseline_z.shape:
            raise ValueError("target_delta_z must have the same shape as baseline_z")
