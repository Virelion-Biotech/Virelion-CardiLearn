"""Common adapters for cardiac omics, ECG/time-series, and imaging inputs."""
from __future__ import annotations
from pathlib import Path
import numpy as np

class OmicsMatrix:
    def __init__(self, matrix: np.ndarray, sample_ids: list[str], feature_names: list[str]):
        if matrix.ndim != 2: raise ValueError("omics matrix must be 2-dimensional")
        if matrix.shape != (len(sample_ids), len(feature_names)): raise ValueError("matrix dimensions do not match IDs/features")
        self.matrix, self.sample_ids, self.feature_names = matrix, sample_ids, feature_names

    @classmethod
    def from_numpy(cls, path: str | Path, sample_ids: list[str], feature_names: list[str]):
        return cls(np.load(path), sample_ids, feature_names)

    def log1p(self): return OmicsMatrix(np.log1p(np.maximum(self.matrix, 0)), self.sample_ids, self.feature_names)
    def zscore(self, axis=0):
        mean = self.matrix.mean(axis=axis, keepdims=True); std = self.matrix.std(axis=axis, keepdims=True); std[std == 0] = 1
        return OmicsMatrix((self.matrix - mean) / std, self.sample_ids, self.feature_names)

class Waveform:
    def __init__(self, values: np.ndarray, sample_rate_hz: float, sample_id: str):
        if values.ndim not in {1, 2}: raise ValueError("waveform must be 1D or 2D [channels, time]")
        if sample_rate_hz <= 0: raise ValueError("sample rate must be positive")
        self.values, self.sample_rate_hz, self.sample_id = values, float(sample_rate_hz), sample_id

    @property
    def duration_s(self) -> float: return self.values.shape[-1] / self.sample_rate_hz

    @classmethod
    def from_numpy(cls, path: str | Path, sample_rate_hz: float, sample_id: str):
        return cls(np.load(path), sample_rate_hz, sample_id)

class ImageTensor:
    def __init__(self, array: np.ndarray, sample_id: str):
        if array.ndim not in {2, 3}: raise ValueError("image must be HxW or HxWxC")
        self.array, self.sample_id = array, sample_id

    def normalize(self):
        x = self.array.astype(np.float32); lo, hi = float(x.min()), float(x.max())
        if hi == lo: return ImageTensor(np.zeros_like(x), self.sample_id)
        return ImageTensor((x - lo) / (hi - lo), self.sample_id)
