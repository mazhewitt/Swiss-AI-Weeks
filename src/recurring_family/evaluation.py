"""Macro-F1 over the fixed label set, per-family F1, the paired bootstrap and the experiment log."""

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import LABELS

LOG_COLUMNS = [
    "date", "row_type", "run_id", "change", "model", "split", "macro_f1", *[f"f1_{l}" for l in LABELS],
    "delta", "verdict", "compared_to", "coverage", "selection_accuracy", "conclusion",
]

TIE_MARGIN = 0.03
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0


def score(truth: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Per-label F1 over all eight labels (a never-predicted label scores 0) plus macro-F1."""
    per_label = f1_score(list(truth), list(predicted), labels=list(LABELS), average=None, zero_division=0)
    result = {"macro_f1": float(per_label.mean())}
    result.update({f"f1_{label}": float(v) for label, v in zip(LABELS, per_label)})
    return result


def _one_hot(labels: pd.Series) -> np.ndarray:
    return (np.asarray(labels, dtype=object)[:, None] == np.array(LABELS, dtype=object)[None, :]).astype(float)


def paired_bootstrap(truth: pd.Series, new: pd.Series, old: pd.Series) -> dict[str, float]:
    """Macro-F1(new) - macro-F1(old) on the same Clients, with a 95% interval from
    resampling Clients (the same resample for both, hence paired)."""
    t, n, o = _one_hot(truth), _one_hot(new), _one_hot(old)
    size = len(t)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(size, size=(BOOTSTRAP_SAMPLES, size))
    weights = np.zeros((BOOTSTRAP_SAMPLES, size))
    np.add.at(weights, (np.arange(BOOTSTRAP_SAMPLES)[:, None], draws), 1.0)

    def macro(w: np.ndarray, pred: np.ndarray) -> np.ndarray:
        tp, true, predicted = w @ (t * pred), w @ t, w @ pred
        denom = true + predicted
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
        return f1.mean(axis=-1)

    delta = float(macro(np.ones(size), n) - macro(np.ones(size), o))
    diffs = macro(weights, n) - macro(weights, o)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"delta": delta, "low": float(low), "high": float(high)}


def verdict(comparison: dict[str, float]) -> str:
    """Deltas under the tie margin are ties; otherwise the interval must exclude 0."""
    if abs(comparison["delta"]) < TIE_MARGIN:
        return "tie"
    if comparison["low"] > 0:
        return "improvement"
    if comparison["high"] < 0:
        return "worse"
    return "tie"


def previous_best(log_path: Path, *, row_type: str, split: str) -> dict | None:
    """The highest-scoring earlier run of the same row type on the same evaluation split."""
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    with open(log_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("row_type") == row_type and r.get("split") == split]
    return max(rows, key=lambda r: float(r["macro_f1"]), default=None)


def log_row(
    scores: dict[str, float], *, change: str, model: str, split: str, conclusion: str,
    row_type: str = "evaluate", run_id: str = "", comparison: dict | None = None,
    diagnostics: dict[str, float] | None = None,
) -> dict:
    row = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_type": row_type,
        "run_id": run_id,
        "change": change,
        "model": model,
        "split": split,
        "conclusion": conclusion,
    }
    row.update({k: f"{v:.4f}" for k, v in scores.items()})
    # a model's own diagnostics (E1: coverage, selection accuracy); blank where undefined
    row.update({k: "" if np.isnan(v) else f"{v:.4f}" for k, v in (diagnostics or {}).items()})
    if comparison is None:
        row["verdict"] = "first"
    else:
        row["delta"] = f"{comparison['delta']:.4f}"
        row["verdict"] = verdict(comparison)
        row["compared_to"] = comparison["run_id"]
    return row


def append_log(log_path: Path, row: dict) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    if not new:
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
            header = reader.fieldnames
        if header != LOG_COLUMNS:  # written by an older version: migrate, never misalign
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
                writer.writeheader()
                for old in existing:
                    writer.writerow({k: old.get(k, "") for k in LOG_COLUMNS})
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})
