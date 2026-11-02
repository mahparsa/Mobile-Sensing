"""Dataset loading."""
import pandas as pd

import config


def load_data(path=config.DATA_PATH):
    """Load the CSV and return features (incl. platform column) and target."""
    df = pd.read_csv(path)

    columns = config.FEATURE_COLUMNS + [config.PLATFORM_COLUMN]
    missing = [c for c in columns + [config.TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")

    X = df[columns].reset_index(drop=True)
    y = df[config.TARGET_COLUMN].reset_index(drop=True)
    return X, y
