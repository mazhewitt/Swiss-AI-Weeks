"""Slow real-data check (Seam 1): the Stream Ranker through the CLI on the fetched challenge data."""

import csv
import time
from pathlib import Path

import pandas as pd
import pytest

from recurring_family.cli import main

REPO = Path(__file__).resolve().parents[1]
REAL_RAW = REPO / "data" / "raw"

# A full evaluation run (stream detection, the fit, 5-fold cross-validation for the decision layer,
# its fit, and scoring the selection set) must fit the evaluation-time budget: about two minutes.
EVALUATION_BUDGET_SECONDS = 120
LABELS = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none"]


def scratch_project(tmp_path: Path) -> list[str]:
    """A scratch project on the real data, so the committed experiment log is never written."""
    if not (REAL_RAW / "valid_labels.csv").exists():
        pytest.skip("real data missing: run `uv run rf fetch-data` first")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "raw").symlink_to(REAL_RAW, target_is_directory=True)
    return ["--root", str(tmp_path)]


@pytest.mark.slow
def test_the_ranker_trains_and_scores_the_selection_set_within_the_evaluation_budget(tmp_path):
    root = scratch_project(tmp_path)
    proba_path = tmp_path / "proba.csv"
    start = time.perf_counter()
    assert main(["train", "--model", "ranker", "--decision", "tuned", *root]) == 0
    assert main([
        "evaluate", "--model", "ranker", "--decision", "tuned", "--proba", str(proba_path),
        "--change", "ranker slow check", *root,
    ]) == 0
    elapsed = time.perf_counter() - start
    assert elapsed <= EVALUATION_BUDGET_SECONDS, f"{elapsed:.0f}s"

    with open(tmp_path / "experiments" / "log.csv", newline="") as f:
        row = list(csv.DictReader(f))[-1]
    assert (row["split"], row["model"]) == ("selection", "ranker+tuned")
    # not a target: a floor that catches a broken pipeline (the prior scores 0.057, E2 0.472)
    assert float(row["macro_f1"]) >= 0.45, row

    proba = pd.read_csv(proba_path, dtype={"client_id": str}).set_index("client_id")
    assert list(proba.columns) == LABELS
    assert len(proba) == 700 and proba.index.is_unique
    assert (proba.sum(axis=1) - 1).abs().max() < 1e-9
