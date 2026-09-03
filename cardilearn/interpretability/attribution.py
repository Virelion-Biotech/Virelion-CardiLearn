"""Attribution methods for the CardiLearn representation model.

The functions here intentionally avoid treating attention weights as explanations.
They measure input-to-output sensitivity with gradients, perturbations, or masking.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import torch


def integrated_gradients(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    *,
    baseline: torch.Tensor | None = None,
    steps: int = 32,
) -> torch.Tensor:
    """Estimate Integrated Gradients for a scalar prediction per sample.

    ``forward_fn`` must return shape ``[batch]``. The returned attribution has the
    same shape as ``inputs`` and approximately satisfies completeness.
    """
    if inputs.ndim < 2:
        raise ValueError("inputs must have shape [batch, ...]")
    if steps < 1:
        raise ValueError("steps must be positive")
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    if baseline.shape != inputs.shape:
        raise ValueError("baseline must have the same shape as inputs")

    original_requires_grad = inputs.requires_grad
    x = inputs.detach()
    b = baseline.detach().to(device=x.device, dtype=x.dtype)
    delta = x - b
    total_grad = torch.zeros_like(x)

    for alpha in torch.linspace(0.0, 1.0, steps, device=x.device, dtype=x.dtype):
        point = (b + alpha * delta).detach().requires_grad_(True)
        output = forward_fn(point)
        if output.ndim != 1 or output.shape[0] != point.shape[0]:
            raise ValueError("forward_fn must return one scalar per input sample")
        gradients = torch.autograd.grad(output.sum(), point, retain_graph=False)[0]
        total_grad += gradients

    attribution = delta * total_grad / steps
    if original_requires_grad:
        inputs.requires_grad_(True)
    return attribution.detach()


def attribute_latent(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    species: torch.Tensor,
    assay: torch.Tensor,
    *,
    latent_index: int,
    baseline: torch.Tensor | None = None,
    steps: int = 32,
) -> torch.Tensor:
    """Attribute a shared latent dimension back to input genes."""
    if latent_index < 0:
        raise ValueError("latent_index must be non-negative")

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        z_shared, *_ = model.encode(x, species, assay)
        if latent_index >= z_shared.shape[1]:
            raise IndexError("latent_index exceeds shared latent dimension")
        return z_shared[:, latent_index]

    return integrated_gradients(forward_fn, inputs, baseline=baseline, steps=steps)


def attribute_prediction(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    *,
    baseline: torch.Tensor | None = None,
    steps: int = 32,
) -> torch.Tensor:
    """Attribute an arbitrary one-score-per-sample prediction to input genes."""
    return integrated_gradients(forward_fn, inputs, baseline=baseline, steps=steps)


def rank_genes(
    attributions: torch.Tensor | np.ndarray,
    gene_names: Sequence[str] | None = None,
    *,
    aggregate: str = "mean_abs",
) -> pd.DataFrame:
    """Rank genes by attribution across a batch.

    ``mean_abs`` highlights magnitude; ``mean_signed`` preserves direction.
    """
    values = np.asarray(attributions.detach().cpu() if torch.is_tensor(attributions) else attributions)
    if values.ndim != 2:
        raise ValueError("attributions must have shape [samples, genes]")
    if aggregate == "mean_abs":
        score = np.mean(np.abs(values), axis=0)
    elif aggregate == "mean_signed":
        score = np.mean(values, axis=0)
    else:
        raise ValueError("aggregate must be 'mean_abs' or 'mean_signed'")
    names = list(gene_names) if gene_names is not None else [f"gene_{i}" for i in range(values.shape[1])]
    if len(names) != values.shape[1]:
        raise ValueError("gene_names length must match the number of genes")
    return pd.DataFrame({"gene": names, "score": score}).sort_values("score", ascending=False).reset_index(drop=True)


def mask_feature_sensitivity(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    *,
    feature_indices: Sequence[int] | None = None,
    baseline_value: float = 0.0,
) -> pd.DataFrame:
    """Measure prediction change after masking individual input features.

    The model is never refit. Positive deltas mean masking reduced the prediction;
    negative deltas mean masking increased it.
    """
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [samples, genes]")
    indices = list(range(inputs.shape[1])) if feature_indices is None else list(feature_indices)
    if any(i < 0 or i >= inputs.shape[1] for i in indices):
        raise IndexError("feature index outside input range")
    with torch.no_grad():
        baseline_prediction = forward_fn(inputs)
        if baseline_prediction.ndim != 1:
            raise ValueError("forward_fn must return one scalar per sample")
        base = float(baseline_prediction.mean().item())
        rows: list[dict[str, float | int]] = []
        for i in indices:
            masked = inputs.clone()
            masked[:, i] = baseline_value
            changed = float(forward_fn(masked).mean().item())
            rows.append({"feature_index": i, "prediction": changed, "delta_from_original": base - changed})
    return pd.DataFrame(rows).sort_values("delta_from_original", ascending=False).reset_index(drop=True)


def permutation_importance(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    *,
    repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance for an unlabeled scalar prediction function.

    Importance is the mean absolute prediction shift caused by independently
    permuting one gene across samples. This is an explanatory sensitivity measure,
    not a held-out predictive score.
    """
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [samples, genes]")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(seed)
    with torch.no_grad():
        original = forward_fn(inputs).detach()
        if original.ndim != 1:
            raise ValueError("forward_fn must return one scalar per sample")
        rows = []
        for gene in range(inputs.shape[1]):
            shifts = []
            for _ in range(repeats):
                permutation = torch.randperm(inputs.shape[0], generator=generator, device=inputs.device)
                permuted = inputs.clone()
                permuted[:, gene] = inputs[permutation, gene]
                shifts.append(torch.mean(torch.abs(original - forward_fn(permuted))).item())
            rows.append({"feature_index": gene, "importance_mean": float(np.mean(shifts)), "importance_std": float(np.std(shifts))})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
