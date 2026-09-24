"""Writing and validating submission CSVs."""

from pathlib import Path

import pandas as pd

from .config import LABELS, PREDICTION_COLUMN

COLUMNS = ["client_id", PREDICTION_COLUMN]


class InvalidSubmission(ValueError):
    pass


def read_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate(submission: pd.DataFrame, expected_clients: pd.Index) -> None:
    """Raise InvalidSubmission unless the table matches the challenge contract."""
    if list(submission.columns) != COLUMNS:
        raise InvalidSubmission(f"columns must be {COLUMNS}, got {list(submission.columns)}")
    ids = submission["client_id"]
    dupes = sorted(set(ids[ids.duplicated()]))
    if dupes:
        raise InvalidSubmission(f"duplicate client_id: {dupes[:5]}")
    missing = sorted(set(expected_clients) - set(ids))
    if missing:
        raise InvalidSubmission(f"{len(missing)} missing client_id, e.g. {missing[:5]}")
    extra = sorted(set(ids) - set(expected_clients))
    if extra:
        raise InvalidSubmission(f"{len(extra)} unexpected client_id, e.g. {extra[:5]}")
    bad = sorted(set(submission[PREDICTION_COLUMN]) - set(LABELS))
    if bad:
        raise InvalidSubmission(f"labels outside the allowed set: {bad[:5]}")


def write_submission(predictions: pd.Series, expected_clients: pd.Index, path: Path) -> None:
    """Write predictions (indexed by client_id) in sample-submission order, then re-validate the file."""
    sub = pd.DataFrame(
        {"client_id": list(expected_clients), PREDICTION_COLUMN: predictions.reindex(expected_clients).tolist()}
    )
    sub[PREDICTION_COLUMN] = sub[PREDICTION_COLUMN].fillna("")
    validate(sub, expected_clients)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    validate(read_submission(path), expected_clients)
