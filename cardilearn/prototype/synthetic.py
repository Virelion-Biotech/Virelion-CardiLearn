"""Deterministic synthetic cardiac-state data for prototype smoke tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    # Fast default: suitable for a first CPU smoke test.
    n_studies: int = 6
    subjects_per_study: int = 2
    samples_per_subject: int = 2
    cells_per_sample: int = 50
    n_genes: int = 256
    n_species: int = 3
    n_assays: int = 2
    seed: int = 42


def generate_synthetic_cardiac_data(config: SyntheticConfig = SyntheticConfig()) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate expression plus hierarchical metadata with known latent biology."""
    rng = np.random.default_rng(config.seed)
    maturity_loading = rng.normal(0, 0.15, config.n_genes)
    injury_loading = rng.normal(0, 0.15, config.n_genes)
    regeneration_loading = rng.normal(0, 0.15, config.n_genes)
    celltype_loading = rng.normal(0, 0.15, (3, config.n_genes))
    maturity_loading[: min(40, config.n_genes)] += 1.0
    maturity_loading[min(40, config.n_genes): min(80, config.n_genes)] -= 1.0
    injury_loading[min(80, config.n_genes): min(110, config.n_genes)] += 1.0
    injury_loading[min(110, config.n_genes): min(140, config.n_genes)] -= 1.0
    regeneration_loading[min(140, config.n_genes): min(170, config.n_genes)] += 1.0
    regeneration_loading[min(170, config.n_genes): min(200, config.n_genes)] -= 1.0
    study_effects = rng.normal(0, 0.45, (config.n_studies, config.n_genes))
    species_effects = rng.normal(0, 0.20, (config.n_species, config.n_genes))
    assay_effects = rng.normal(0, 0.15, (config.n_assays, config.n_genes))
    rows: list[dict[str, object]] = []
    expression: list[np.ndarray] = []
    cell_counter = 0
    for study in range(config.n_studies):
        study_species = study % config.n_species
        study_assay = study % config.n_assays
        for subject in range(config.subjects_per_study):
            subject_id = f"ST{study:02d}_SUB{subject:02d}"
            maturity_shift = rng.normal(0, 0.08)
            injury_shift = rng.normal(0, 0.05)
            for sample in range(config.samples_per_subject):
                sample_id = f"ST{study:02d}_SUB{subject:02d}_SAMP{sample:02d}"
                maturity_base = float(np.clip(0.15 + 0.70 * (sample / max(config.samples_per_subject - 1, 1)) + rng.normal(0, 0.05) + maturity_shift, 0, 1))
                injured = int(rng.random() < 0.35)
                regeneration_base = 0.75 if injured and maturity_base < 0.55 and study % 2 == 0 else 0.25
                regeneration_base = float(np.clip(regeneration_base + rng.normal(0, 0.08), 0, 1))
                injury_base = float(np.clip((1.0 if injured else 0.0) + injury_shift, 0, 1))
                for _ in range(config.cells_per_sample):
                    ct = int(rng.choice(3, p=[0.60, 0.20, 0.20]))
                    cell_maturity = maturity_base if ct == 0 else rng.uniform(0.05, 0.30)
                    cell_injury = injury_base * (1.0 if ct == 0 else 0.5)
                    cell_regeneration = regeneration_base if ct == 0 else regeneration_base * 0.4
                    signal = (
                        cell_maturity * maturity_loading
                        + cell_injury * injury_loading
                        + cell_regeneration * regeneration_loading
                        + celltype_loading[ct]
                        + study_effects[study]
                        + species_effects[study_species]
                        + assay_effects[study_assay]
                        + rng.normal(0, 0.35, config.n_genes)
                    )
                    expression.append(np.maximum(signal + 2.0, 0).astype(np.float32))
                    rows.append({
                        "study_id": f"ST{study:02d}",
                        "subject_id": subject_id,
                        "sample_id": sample_id,
                        "cell_id": f"C{cell_counter:07d}",
                        "species": study_species,
                        "assay": study_assay,
                        "cell_type": ct,
                        "maturation": float(cell_maturity),
                        "injury": float(cell_injury > 0.35),
                        "regeneration": float(cell_regeneration),
                    })
                    cell_counter += 1
    return np.vstack(expression), pd.DataFrame(rows)
