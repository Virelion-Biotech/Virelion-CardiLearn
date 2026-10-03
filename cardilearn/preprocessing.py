"""Train-only preprocessing for numeric and categorical feature tables."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing graph from feature dtypes without fitting it.

    Fitting is intentionally left to the training pipeline so validation/test data can
    never influence imputation, scaling, or category discovery.
    """

    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric:
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        transformers.append(("categorical", categorical_pipeline, categorical))
    if not transformers:
        raise ValueError("no usable feature columns found")

    return ColumnTransformer(transformers=transformers, remainder="drop")
