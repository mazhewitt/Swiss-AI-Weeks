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
    "delta", "verdict", "compared_to", "coverage", "selection_accuracy",
    # what the model was trained on: all Clients (real-labelled plus Pseudo-Labelled), the Pseudo-Label
    # sources (split@Shifted Cutoff) and their settings, and the latest fidelity check when it was trained
    "training_clients", "pseudo_sources", "pseudo_weight", "pseudo_min_payments", "pseudo_min_payments_before",
    "pseudo_churn", "fidelity",
    "conclusion",
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


def _resample_weights(size: int) -> np.ndarray:
    """How often each Client is drawn in each bootstrap resample: the same fixed-seed resamples for
    every call on the same number of Clients, so any two runs scored on them are paired."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(size, size=(BOOTSTRAP_SAMPLES, size))
    weights = np.zeros((BOOTSTRAP_SAMPLES, size))
    np.add.at(weights, (np.arange(BOOTSTRAP_SAMPLES)[:, None], draws), 1.0)
    return weights


def _weighted_macro_f1(weights: np.ndarray, truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    tp, true, pred = weights @ (truth * predicted), weights @ truth, weights @ predicted
    denom = true + pred
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    return f1.mean(axis=-1)


def paired_bootstrap(truth: pd.Series, new: pd.Series, old: pd.Series) -> dict[str, float]:
    """Macro-F1(new) - macro-F1(old) on the same Clients, with a 95% interval from
    resampling Clients (the same resample for both, hence paired)."""
    t, n, o = _one_hot(truth), _one_hot(new), _one_hot(old)
    size = len(t)
    weights = _resample_weights(size)
    delta = float(_weighted_macro_f1(np.ones(size), t, n) - _weighted_macro_f1(np.ones(size), t, o))
    diffs = _weighted_macro_f1(weights, t, n) - _weighted_macro_f1(weights, t, o)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"delta": delta, "low": float(low), "high": float(high)}


def bootstrap_interval(truth: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Macro-F1 of one run with a 95% interval over the same Client resamples `paired_bootstrap` uses."""
    t, p = _one_hot(truth), _one_hot(predicted)
    scores = _weighted_macro_f1(_resample_weights(len(t)), t, p)
    low, high = np.percentile(scores, [2.5, 97.5])
    return {"macro_f1": float(_weighted_macro_f1(np.ones(len(t)), t, p)), "low": float(low), "high": float(high)}


def verdict(comparison: dict[str, float]) -> str:
    """Deltas under the tie margin are ties; otherwise the interval must exclude 0."""
    if abs(comparison["delta"]) < TIE_MARGIN:
        return "tie"
    if comparison["low"] > 0:
        return "improvement"
    if comparison["high"] < 0:
        return "worse"
    return "tie"


def latest_fidelity(log_path: Path) -> dict | None:
    """The latest Pseudo-Label fidelity check in the log, {run_id, verdict}, or None if none is logged."""
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    with open(log_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("row_type") == "fidelity"]
    return {"run_id": rows[-1]["run_id"], "verdict": rows[-1]["verdict"]} if rows else None


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
