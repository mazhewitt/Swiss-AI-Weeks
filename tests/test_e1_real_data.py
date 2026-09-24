"""Slow real-data checks (Seam 1): the rules model through the CLI on the fetched challenge data."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from recurring_family.cli import main

REPO = Path(__file__).resolve().parents[1]
REAL_RAW = REPO / "data" / "raw"
MILESTONE2 = REPO / "submissions" / "milestone2_rules_none_gate_v2.csv"

# E1 (gate off) on the ~700-Client selection set with the current stream detector (ranker spec,
# story 4). The band guards against detector regressions, not against an out-of-date number.
REFERENCE_MACRO_F1 = 0.5365
TOLERANCE = 0.02


def scratch_project(tmp_path: Path) -> list[str]:
    """A scratch project on the real data, so the committed experiment log is never written."""
    if not (REAL_RAW / "valid_labels.csv").exists():
        pytest.skip("real data missing: run `uv run rf fetch-data` first")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "raw").symlink_to(REAL_RAW, target_is_directory=True)
    return ["--root", str(tmp_path)]


@pytest.mark.slow
def test_e1_macro_f1_on_the_valid_selection_set_is_near_the_reference(tmp_path):
    root = scratch_project(tmp_path)
    assert main(["train", "--model", "rules", *root]) == 0
    assert main(["evaluate", "--model", "rules", "--change", "E1 slow check", *root]) == 0

    with open(tmp_path / "experiments" / "log.csv", newline="") as f:
        row = list(csv.DictReader(f))[-1]
    assert (row["split"], row["model"]) == ("selection", "rules")
    assert float(row["macro_f1"]) == pytest.approx(REFERENCE_MACRO_F1, abs=TOLERANCE)


@pytest.mark.slow
def test_gated_rules_reproduce_the_milestone2_submission(tmp_path):
    root = scratch_project(tmp_path)
    assert main(["train", "--model", "rules", "--none-gate", *root]) == 0
    assert main(["submit", "--model", "rules", "--name", "m2", *root]) == 0

    new = pd.read_csv(tmp_path / "submissions" / "m2.csv", dtype=str, keep_default_na=False)
    committed = pd.read_csv(MILESTONE2, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(new, committed)
