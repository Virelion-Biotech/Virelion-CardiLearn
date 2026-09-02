"""CardiLearn v0.1 research prototype.

This package contains the new cardiac state representation model described in
`docs/CARDILEARN_MODEL_V0_1.md`. It is intentionally separate from the older
benchmark orchestration code so the model can evolve without changing the
CardiAtlas/CardiBench contracts.
"""

from .model import CardiLearnProto

__all__ = ["CardiLearnProto"]
