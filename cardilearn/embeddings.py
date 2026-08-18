"""Representation-learning interfaces with a dependency-light fallback."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class TabularEmbedder:
    """Stable PCA embedding baseline; deep encoders can implement the same interface later."""
    def __init__(self, n_components: int = 32) -> None:
        self.n_components = n_components
        self.pipeline = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=n_components, random_state=42))])

    def fit(self, X): self.pipeline.fit(X); return self
    def transform(self, X) -> np.ndarray: return self.pipeline.transform(X)
    def fit_transform(self, X) -> np.ndarray: return self.pipeline.fit_transform(X)
    def save_metadata(self) -> dict: return {"type": "pca", "n_components": self.n_components}
