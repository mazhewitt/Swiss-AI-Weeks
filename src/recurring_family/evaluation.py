"""Macro-F1 over the fixed label set, per-family F1 and the experiment log."""

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from .config import LABELS

LOG_COLUMNS = ["date", "change", "model", "split", "macro_f1", *[f"f1_{l}" for l in LABELS], "conclusion"]


def score(truth: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Per-label F1 over all eight labels (a never-predicted label scores 0) plus macro-F1."""
    per_label = f1_score(list(truth), list(predicted), labels=list(LABELS), average=None, zero_division=0)
    result = {"macro_f1": float(per_label.mean())}
    result.update({f"f1_{label}": float(v) for label, v in zip(LABELS, per_label)})
    return result


def log_row(scores: dict[str, float], *, change: str, model: str, split: str, conclusion: str) -> dict:
    row = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "change": change,
        "model": model,
        "split": split,
        "conclusion": conclusion,
    }
    row.update({k: f"{v:.4f}" for k, v in scores.items()})
    return row


def append_log(log_path: Path, row: dict) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})
