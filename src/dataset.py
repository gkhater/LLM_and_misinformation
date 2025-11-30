"""Dataset loading and filtering utilities."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ("id", "claim")


def load_dataset(config: dict) -> pd.DataFrame:
    """Load dataset and enforce expected columns and optional split filtering."""
    path = config["dataset"]["csv_path"]
    df = pd.read_csv(path)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    split_col = config["dataset"].get("split_column")
    split_filter = config["dataset"].get("split_filter")
    if split_col and split_filter:
        if split_col not in df.columns:
            raise ValueError(f"Configured split_column '{split_col}' not in dataset.")
        df = df[df[split_col] == split_filter].reset_index(drop=True)

    return df
