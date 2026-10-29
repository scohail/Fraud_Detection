"""Data loading and splitting."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Class"
RANDOM_STATE = 42

# Project root = parent of this package; the dataset lives in <root>/data/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = PROJECT_ROOT / "data" / "creditcard.csv"


def load_data(path=DEFAULT_PATH, drop_duplicates=True):
    """Load the credit-card transactions dataset.

    Parameters
    ----------
    path : str or Path
        Location of ``creditcard.csv`` (Kaggle "Credit Card Fraud Detection").
    drop_duplicates : bool
        The raw file contains ~1 000 exact duplicate rows. Keeping them lets the
        same transaction land in both the training and the test side of a split,
        which slightly inflates scores, so they are removed by default.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it in data/."
        )
    df = pd.read_csv(path)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    return df


def split_features_target(df):
    """Return ``(X, y)`` from a dataframe containing the ``Class`` column."""
    return df.drop(columns=TARGET), df[TARGET]


def train_test_split_stratified(df, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified hold-out split used by every notebook and script.

    The test set is put aside once and only used for the final evaluation;
    all model selection happens with cross-validation on the training set.
    """
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
