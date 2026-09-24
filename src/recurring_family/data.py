"""Typed loaders for the challenge files in the raw data area."""

from pathlib import Path

import pandas as pd

from .config import LABEL_COLUMN, PREDICTION_COLUMN

SPLITS = ("train", "valid", "test", "unlabeled")
LABELLED_SPLITS = ("train", "valid")

_TRANSACTION_FILES = {
    "train": "train_transactions.jsonl",
    "valid": "valid_transactions.jsonl",
    "test": "test_transactions.jsonl",
    "unlabeled": "unlabeled_pretrain_transactions.jsonl",
}

TRANSACTION_DTYPES = {
    "client_id": "string",
    "amount": "float64",
    "currency": "string",
    "direction": "string",
    "type": "string",
    "mcc": "string",
    "description": "string",
    "fee": "float64",
}


def _check_split(split: str, allowed: tuple[str, ...]) -> None:
    if split not in allowed:
        raise ValueError(f"unknown split {split!r}; expected one of {allowed}")


def load_transactions(raw_dir: Path, split: str) -> pd.DataFrame:
    """One row per transaction, `timestamp` as tz-aware UTC, `mcc` as a string."""
    _check_split(split, SPLITS)
    df = pd.read_json(
        Path(raw_dir) / _TRANSACTION_FILES[split],
        lines=True,
        dtype={"mcc": str},
        convert_dates=False,
    )
    df = df.astype(TRANSACTION_DTYPES)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_labels(raw_dir: Path, split: str) -> pd.DataFrame:
    """Columns: client_id, cutoff_date (UTC), target_next_recurring_merchant."""
    _check_split(split, LABELLED_SPLITS)
    df = pd.read_csv(
        Path(raw_dir) / f"{split}_labels.csv",
        dtype={"client_id": "string", LABEL_COLUMN: "string"},
        keep_default_na=False,
    )
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"], utc=True)
    return df


def load_sample_submission(raw_dir: Path) -> pd.DataFrame:
    """The exact test Client set: columns client_id, predicted_next_recurring_merchant."""
    return pd.read_csv(
        Path(raw_dir) / "sample_submission.csv",
        dtype={"client_id": "string", PREDICTION_COLUMN: "string"},
        keep_default_na=False,
    )
