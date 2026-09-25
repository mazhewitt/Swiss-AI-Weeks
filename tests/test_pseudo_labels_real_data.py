"""Slow real-data check (Seam 1): the Pseudo-Label fidelity check on the real train split."""

import csv
import re
from pathlib import Path

import pytest

from recurring_family.cli import main
from recurring_family.pseudo import (
    FIDELITY_BEFORE_CANDIDATES, FIDELITY_CANDIDATES, FIDELITY_CHURN_CANDIDATES, PSEUDO_CHURN, PSEUDO_MIN_PAYMENTS,
    PSEUDO_MIN_PAYMENTS_BEFORE,
)

REPO = Path(__file__).resolve().parents[1]
REAL_RAW = REPO / "data" / "raw"

# What the check found on the real train split at the default Shifted Cutoff (2025-10-03), choosing
# among min_payments 2-10, min_payments_before 0-3 and churn 0-0.3 (ticket 08). The real task: none
# share 0.2985, milestone-2 rule macro-F1 0.5331. Without churn no setting brings both within 0.05
# (ticket 04: the none share reaches the real one only at 7 payments, where the rule scores 0.78):
# streams inside the known history almost never stop, but at the real Cutoff many do. Churning a
# fifth of the Clients at the Shifted Cutoff closes both gaps.
# Ticket 11 let stray Filler Description payments join a stream on its schedule. That changes the
# streams the rule and the labeller see, so the figures moved (the choice did not): the real rule
# macro-F1 0.5331 -> 0.5245, the chosen setting's none share 0.3035 -> 0.3005 and its rule macro-F1
# 0.5433 -> 0.5402 (fidelity runs 20260924T214629-add1e5 before, 20260925T011208-e56725 after).
# Ticket 12 let Decoy-described payments join a stream on its schedule (`join_decoys`), which moved them
# again (the choice did not): real rule macro-F1 0.5245 -> 0.5236, none share 0.3005 -> 0.3000, rule
# macro-F1 0.5402 -> 0.5380.
CHOSEN = "min_payments 3, min_payments_before 0, churn 0.2"
REAL_NONE_SHARE = 0.2985
REAL_RULE_MACRO_F1 = 0.5236
PSEUDO_NONE_SHARE = 0.3000
PSEUDO_RULE_MACRO_F1 = 0.5380


@pytest.mark.slow
def test_the_fidelity_check_on_train_chooses_and_records_the_labeller_settings(tmp_path, capsys):
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
    assert f"chosen {CHOSEN}" in out
    assert f"{CHOSEN} (chosen from min_payments 2-10 x min_payments_before 0-3 x churn 0,0.1,0.15,0.2,0.25,0.3" in (
        row["change"]
    )
    assert FIDELITY_CANDIDATES == tuple(range(2, 11)) and FIDELITY_BEFORE_CANDIDATES == (0, 1, 2, 3)
    assert FIDELITY_CHURN_CANDIDATES == (0.0, 0.1, 0.15, 0.2, 0.25, 0.3)
    # the defaults stay ticket 04's: the ranker trained on the chosen Pseudo-Labels only tied it on train
    assert (PSEUDO_MIN_PAYMENTS, PSEUDO_MIN_PAYMENTS_BEFORE, PSEUDO_CHURN) == (4, 0, 0.0)

    assert f"vs real {REAL_NONE_SHARE:.4f}" in out
    assert f"vs real {REAL_RULE_MACRO_F1:.4f}" in out
    assert float(row["macro_f1"]) == pytest.approx(PSEUDO_RULE_MACRO_F1, abs=0.005)
    assert float(row["delta"]) == pytest.approx(PSEUDO_RULE_MACRO_F1 - REAL_RULE_MACRO_F1, abs=0.005)
    none_share = float(re.search(r"none share ([0-9.]+) vs real", row["conclusion"]).group(1))
    assert none_share == pytest.approx(PSEUDO_NONE_SHARE, abs=0.005)
    assert row["verdict"] == "pass" and out.rstrip().endswith("PASS")

    kept = tmp_path / "experiments" / "fidelity" / f"{row['run_id']}.csv"
    assert sum(1 for _ in open(kept)) == 198 + 1  # every setting's verdict, plus the header

    table = tmp_path / "artifacts" / "pseudo_labels" / "train-2025-10-03-min3-churn0.2.csv"
    assert sum(1 for _ in open(table)) == 2000 + 1  # one row per train Client, plus the header
