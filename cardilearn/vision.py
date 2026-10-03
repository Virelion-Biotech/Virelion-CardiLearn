"""Image-to-feature utilities for cardiac imaging experiments."""
from __future__ import annotations

import numpy as np


def image_statistics(image) -> dict[str, float]:
    """Return simple, reproducible image statistics; advanced CNNs can consume the same IDs."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim not in {2, 3}:
        raise ValueError("image must be 2-D grayscale or 3-D channel-last")
    if array.ndim == 3:
        channel_mean = array.mean(axis=(0, 1))
        channel_std = array.std(axis=(0, 1))
        out = {
            "height": float(array.shape[0]),
            "width": float(array.shape[1]),
            "channels": float(array.shape[2]),
            "global_mean": float(array.mean()),
            "global_std": float(array.std()),
        }
        for i, (mean, std) in enumerate(zip(channel_mean, channel_std)):
            out[f"channel_{i}_mean"] = float(mean)
            out[f"channel_{i}_std"] = float(std)
        return out
    return {
        "height": float(array.shape[0]),
        "width": float(array.shape[1]),
        "channels": 1.0,
        "global_mean": float(array.mean()),
        "global_std": float(array.std()),
    }


def load_pil(path):
    """Load an image only when the optional Pillow dependency is installed."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("install the 'vision' extra to use image loading") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))
