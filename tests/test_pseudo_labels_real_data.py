"""Slow real-data check (Seam 1): the Pseudo-Label fidelity check on the real train split."""

import csv
import re
from pathlib import Path

import pytest

from recurring_family.cli import main
from recurring_family.pseudo import FIDELITY_CANDIDATES, PSEUDO_MIN_PAYMENTS

REPO = Path(__file__).resolve().parents[1]
REAL_RAW = REPO / "data" / "raw"

# What the check found on the real train split at the default Shifted Cutoff (2025-10-03), choosing
# among min_payments 2-10. The real task: none share 0.2985, milestone-2 rule macro-F1 0.5331. No
# setting brings both within 0.05: the none share reaches the real one only at 7 payments, where the
# rule scores 0.78 against Pseudo-Labels. The chosen setting has the smallest worse gap.
CHOSEN_MIN_PAYMENTS = 4
REAL_NONE_SHARE = 0.2985
REAL_RULE_MACRO_F1 = 0.5331
PSEUDO_NONE_SHARE = 0.1535
PSEUDO_RULE_MACRO_F1 = 0.6453


@pytest.mark.slow
def test_the_fidelity_check_on_train_chooses_and_records_min_payments(tmp_path, capsys):
    if not (REAL_RAW / "train_labels.csv").exists():
        pytest.skip("real data missing: run `uv run rf fetch-data` first")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "raw").symlink_to(REAL_RAW, target_is_directory=True)
    root = ["--root", str(tmp_path)]  # a scratch project, so the committed log is never written

    assert main(["pseudo-labels", "--split", "train", "--fidelity", *root]) == 0
    out = capsys.readouterr().out
    print(out)

    with open(tmp_path / "experiments" / "log.csv", newline="") as f:
        [row] = list(csv.DictReader(f))
    assert (row["row_type"], row["split"], row["model"]) == ("fidelity", "train", "rules+gate4")
    assert f"chosen min_payments {CHOSEN_MIN_PAYMENTS}" in out
    assert f"min_payments {CHOSEN_MIN_PAYMENTS} (chosen from 2-10)" in row["change"]
    # the command's default is the setting the check chose
    assert PSEUDO_MIN_PAYMENTS == CHOSEN_MIN_PAYMENTS and FIDELITY_CANDIDATES == tuple(range(2, 11))

    assert f"vs real {REAL_NONE_SHARE:.4f}" in out
    assert f"vs real {REAL_RULE_MACRO_F1:.4f}" in out
    assert float(row["macro_f1"]) == pytest.approx(PSEUDO_RULE_MACRO_F1, abs=0.005)
    assert float(row["delta"]) == pytest.approx(PSEUDO_RULE_MACRO_F1 - REAL_RULE_MACRO_F1, abs=0.005)
    none_share = float(re.search(r"none share ([0-9.]+) vs real", row["conclusion"]).group(1))
    assert none_share == pytest.approx(PSEUDO_NONE_SHARE, abs=0.005)
    assert row["verdict"] == "fail" and out.rstrip().endswith("FAIL")

    table = tmp_path / "artifacts" / "pseudo_labels" / f"train-2025-10-03-min{CHOSEN_MIN_PAYMENTS}.csv"
    assert sum(1 for _ in open(table)) == 2000 + 1  # one row per train Client, plus the header
