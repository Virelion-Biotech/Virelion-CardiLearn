"""CardiLearn v0.1 research prototype.

Torch-backed model components are imported lazily so metadata, leakage, and
other non-training utilities remain usable without installing Torch.
"""

__all__ = ["CardiLearnProto"]


def __getattr__(name: str):
    if name == "CardiLearnProto":
        from .model import CardiLearnProto
        return CardiLearnProto
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
