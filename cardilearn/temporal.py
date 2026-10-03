"""Forward-chaining validation for longitudinal cardiac experiments."""
from __future__ import annotations
import pandas as pd

def forward_split(df: pd.DataFrame, time_column: str, *, validation_fraction: float = 0.2, test_fraction: float = 0.2):
    if time_column not in df: raise ValueError(f"missing time column: {time_column}")
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1: raise ValueError("invalid split fractions")
    order = df.sort_values(time_column, kind="stable")
    n = len(order); test_n = max(1, int(round(n * test_fraction))); val_n = max(1, int(round(n * validation_fraction)))
    train = order.iloc[: n - val_n - test_n]; validation = order.iloc[n - val_n - test_n : n - test_n]; test = order.iloc[n - test_n :]
    return train.copy(), validation.copy(), test.copy()
