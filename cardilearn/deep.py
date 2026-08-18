"""Optional PyTorch models for dense cardiac representations."""
from __future__ import annotations
try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    torch = None; nn = None; _IMPORT_ERROR = exc

def _require_torch():
    if torch is None: raise ImportError("PyTorch is optional; install `virelion-cardilearn[torch]` to use deep models") from _IMPORT_ERROR

if nn is not None:
    class MLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden=(256, 128), dropout: float = .1):
            super().__init__(); layers=[]; last=input_dim
            for width in hidden: layers += [nn.Linear(last, width), nn.ReLU(), nn.Dropout(dropout)]; last=width
            layers.append(nn.Linear(last, output_dim)); self.network=nn.Sequential(*layers)
        def forward(self, x): return self.network(x)

def make_mlp(input_dim: int, output_dim: int, **kwargs):
    _require_torch(); return MLP(input_dim, output_dim, **kwargs)
