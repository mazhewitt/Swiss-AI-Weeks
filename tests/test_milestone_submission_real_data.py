"""Slow real-data checks (Seam 1): the committed Day-2 12:00 milestone submission is valid and reproducible."""

from pathlib import Path

import pandas as pd
import pytest

from recurring_family.cli import main

REPO = Path(__file__).resolve().parents[1]
REAL_RAW = REPO / "data" / "raw"
NAME = "day2_noon_rules_gate4"
SUBMISSION = REPO / "submissions" / f"{NAME}.csv"
COMPARISON = REPO / "submissions" / f"{NAME}.md"
LABELS = {"cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none"}


def scratch_project(tmp_path: Path) -> list[str]:
    if not (REAL_RAW / "sample_submission.csv").exists():
        pytest.skip("real data missing: run `uv run rf fetch-data` first")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "raw").symlink_to(REAL_RAW, target_is_directory=True)
    return ["--root", str(tmp_path)]


@pytest.mark.slow
def test_day2_noon_submission_is_valid_and_has_its_comparison_beside_it(tmp_path, capsys):
    root = scratch_project(tmp_path)
    assert main(["submit", "--check", str(SUBMISSION), *root]) == 0
    sub = pd.read_csv(SUBMISSION, dtype=str, keep_default_na=False)
    assert len(sub) == 1000 and sub["client_id"].is_unique
    assert set(sub["predicted_next_recurring_merchant"]) <= LABELS
    text = COMPARISON.read_text()
    assert "Winner:" in text and "95% interval" in text and "day2_noon_milestone.sh" in text


@pytest.mark.slow
def test_day2_noon_submission_is_reproduced_by_its_committed_commands(tmp_path):
    root = scratch_project(tmp_path)
    assert main(["train", "--model", "rules", "--none-gate", "--with-selection", *root]) == 0
    assert main(["submit", "--model", "rules", "--name", NAME, *root]) == 0
    assert (tmp_path / "submissions" / f"{NAME}.csv").read_bytes() == SUBMISSION.read_bytes()
