"""Slow real-data check (Seam 1): E1 through the CLI on the fetched challenge data."""

import csv
from pathlib import Path

import pytest

from recurring_family.cli import main

REAL_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# E1's macro-F1 on the valid selection set with the ticket 03/08 stream detector (coverage 0.7111,
# selection accuracy 0.7955). The spec's reference, 0.479, is the fact-finding detector's score on all
# 1,000 valid Clients (research/data-facts.md); that detector scores 0.4881 with the same rule on these
# same Clients. The gap comes from the better detector (E1 on train rises too: 0.485 -> 0.5043), so the
# check is anchored on this pipeline's own number, keeping the spec's +/-0.04 band both ways: a drop
# means stream detection regressed, a jump means something to distrust (e.g. leakage).
REFERENCE_MACRO_F1 = 0.5365
TOLERANCE = 0.04


@pytest.mark.slow
def test_e1_macro_f1_on_the_valid_selection_set_is_near_the_reference(tmp_path):
    if not (REAL_RAW / "valid_labels.csv").exists():
        pytest.skip("real data missing: run `uv run rf fetch-data` first")
    # a scratch project on the real data, so the committed experiment log is never written
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "raw").symlink_to(REAL_RAW, target_is_directory=True)
    root = ["--root", str(tmp_path)]

    assert main(["train", "--model", "rules", *root]) == 0
    assert main(["evaluate", "--model", "rules", "--change", "E1 slow check", *root]) == 0

    with open(tmp_path / "experiments" / "log.csv", newline="") as f:
        row = list(csv.DictReader(f))[-1]
    assert row["split"] == "selection"
    assert float(row["macro_f1"]) == pytest.approx(REFERENCE_MACRO_F1, abs=TOLERANCE)
