"""Typed loaders for the challenge files in the raw data area."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABEL_COLUMN, PREDICTION_COLUMN

SPLITS = ("train", "valid", "test", "unlabeled")
LABELLED_SPLITS = ("train", "valid")
# valid is divided once into these two disjoint sets of Clients.
VALID_SETS = ("selection", "holdout")
LABEL_SOURCES = (*LABELLED_SPLITS, *VALID_SETS)

HOLDOUT_FRACTION = 0.3
SPLIT_SEED = 20260101

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


class DataError(ValueError):
    """A raw file does not have the shape the loaders promise."""


# An explicit UTC designator or offset; naive timestamps are ambiguous and rejected.
_TZ_SUFFIX = r"(?:Z|[+-]\d{2}:?\d{2})$"


def _parse_utc_timestamps(raw: pd.Series, split: str) -> pd.Series:
    text = raw.astype("string")
    bad = text.isna() | ~text.str.contains(_TZ_SUFFIX, regex=True).fillna(False)
    if not bad.any():
        parsed = pd.to_datetime(text, utc=True, errors="coerce")
        bad = parsed.isna()
    if bad.any():
        row = int(bad.to_numpy().nonzero()[0][0])
        raise DataError(
            f"{split} transactions: {int(bad.sum())} row(s) with a missing, malformed or naive "
            f"timestamp (first at row {row}: {raw.iloc[row]!r}); expected ISO 8601 with Z or an offset"
        )
    return parsed


class LabelLeak(DataError):
    """Labels were read where the evaluation protocol forbids it."""


@dataclass(frozen=True)
class _LabelPolicy:
    training: bool = False
    selection: bool = True
    holdout: bool = False


_POLICY: ContextVar[_LabelPolicy] = ContextVar("label_policy", default=_LabelPolicy())


@contextmanager
def _policy(policy: _LabelPolicy):
    token = _POLICY.set(policy)
    try:
        yield
    finally:
        _POLICY.reset(token)


def training_run(*, with_selection: bool = False):
    """Inside a training run valid labels are off limits, except the selection set when refitting."""
    return _policy(_LabelPolicy(training=True, selection=with_selection, holdout=False))


def checkpoint():
    """The human-started checkpoint: the only place sealed-holdout labels may be read."""
    return _policy(_LabelPolicy(training=False, selection=True, holdout=True))


def _guard_label_read(source: str) -> None:
    if source == "train":
        return
    policy = _POLICY.get()
    if policy.training and (source != "selection" or not policy.selection):
        raise LabelLeak(
            f"a training run tried to read valid labels ({source}); training uses train labels only"
            + ("" if source != "selection" else " (pass --with-selection to refit on train plus the selection set)")
        )
    if source in ("holdout", "valid") and not policy.holdout:
        raise LabelLeak(
            f"reading {source} labels would expose the sealed holdout; "
            "sealed holdout labels are read only in checkpoint mode (evaluate --checkpoint)"
        )


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
    df["timestamp"] = _parse_utc_timestamps(df["timestamp"], split)
    return df


def _read_label_file(raw_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(
        Path(raw_dir) / f"{split}_labels.csv",
        dtype={"client_id": "string", LABEL_COLUMN: "string"},
        keep_default_na=False,
    )
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"], utc=True)
    return df


def load_labels(raw_dir: Path, split: str) -> pd.DataFrame:
    """Columns: client_id, cutoff_date (UTC), target_next_recurring_merchant.

    `split` is train, valid (all of it), or one of the valid sets: selection or holdout.
    Guarded: no valid labels inside a training run (bar the selection set when refitting),
    and no holdout labels (hence no full valid) outside checkpoint mode.
    """
    _check_split(split, LABEL_SOURCES)
    _guard_label_read(split)
    if split in LABELLED_SPLITS:
        return _read_label_file(raw_dir, split)
    df = _read_label_file(raw_dir, "valid")
    members = valid_split(raw_dir).query("set == @split")["client_id"]
    return df[df["client_id"].isin(set(members))].reset_index(drop=True)


def valid_split(raw_dir: Path) -> pd.DataFrame:
    """Columns client_id, set: every valid Client in exactly one of selection / holdout.

    Stratified by label with a fixed seed. Within each label the Clients are ordered by
    ID before shuffling, so the split depends only on the (client, label) pairs, never on
    file order. The holdout takes round(30%) of valid; per-label quotas use largest
    remainder so every label gets floor or ceil of its 30% share. Rows keep file order.
    """
    df = _read_label_file(raw_dir, "valid")
    counts = df[LABEL_COLUMN].value_counts().sort_index()
    exact = counts * HOLDOUT_FRACTION
    quota = np.floor(exact).astype(int)
    extra = int(round(len(df) * HOLDOUT_FRACTION)) - int(quota.sum())
    remainders = (exact - quota).sort_values(ascending=False, kind="stable")
    for label in remainders.index[:extra]:
        quota[label] += 1

    rng = np.random.default_rng(SPLIT_SEED)
    holdout: set[str] = set()
    for label, k in quota.items():
        ids = np.array(sorted(df.loc[df[LABEL_COLUMN] == label, "client_id"]), dtype=object)
        holdout.update(rng.permutation(ids)[:k])
    sets = np.where(df["client_id"].isin(holdout), "holdout", "selection")
    return pd.DataFrame({"client_id": df["client_id"].astype(str), "set": sets})


def load_sample_submission(raw_dir: Path) -> pd.DataFrame:
    """The exact test Client set: columns client_id, predicted_next_recurring_merchant."""
    return pd.read_csv(
        Path(raw_dir) / "sample_submission.csv",
        dtype={"client_id": "string", PREDICTION_COLUMN: "string"},
        keep_default_na=False,
    )
