"""CardiLearn Step 12 ablation framework.

Ablations alter one scientific assumption while preserving:
- dataset
- frozen split
- seed
- encoder capacity
- optimizer
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AblationConfig:
    reconstruction: bool = True
    contrastive: bool = True
    transition: bool = True
    species_adversary: bool = True
    metadata: bool = True
    latent_dim: int = 128
    metadata_shuffle: bool = False


def registry():
    base = AblationConfig()
    return {
        "full": base,
        "no_reconstruction": replace(base, reconstruction=False),
        "no_contrastive": replace(base, contrastive=False),
        "no_species_adversary": replace(base, species_adversary=False),
        "no_transition": replace(base, transition=False),
        "no_metadata": replace(base, metadata=False),
        "randomized_metadata": replace(base, metadata_shuffle=True),
        "no_latent_bottleneck": replace(base, latent_dim=4096),
    }


def validate_pair(reference: AblationConfig, variant: AblationConfig):
    """Ensure future ablations modify only declared assumptions."""
    changed = sum([
        reference.reconstruction != variant.reconstruction,
        reference.contrastive != variant.contrastive,
        reference.transition != variant.transition,
        reference.species_adversary != variant.species_adversary,
        reference.metadata != variant.metadata,
        reference.latent_dim != variant.latent_dim,
        reference.metadata_shuffle != variant.metadata_shuffle,
    ])
    if changed != 1:
        raise ValueError("Invalid ablation: more than one component changed")
