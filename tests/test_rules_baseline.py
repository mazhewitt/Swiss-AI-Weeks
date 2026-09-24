"""Seam 1: the E1 rule baseline (`--model rules`) through the CLI against fixture data."""

import csv
import json

import pandas as pd
import pytest

LABELS = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none"]
TEST_IDS = ["C000004", "C000008", "C000010"]
DAY = pd.Timedelta(days=1)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def payments(client, description, mcc, amount, last, n, every_days=30):
    """`n` outgoing card payments every `every_days` days, the last one on `last` (before the Cutoff)."""
    end = pd.Timestamp(last, tz="UTC") + pd.Timedelta(hours=10)
    return [
        {
            "amount": amount, "client_id": client, "currency": "eur", "description": description,
            "direction": "out", "fee": 0.0, "mcc": mcc,
            "timestamp": (end - (n - 1 - i) * every_days * DAY).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": "card_payment",
        }
        for i in range(n)
    ]


def write_train(project, transactions, labels):
    """Replace the train split: its transactions and its labels (Clients may have no transactions)."""
    with open(project.raw / "train_transactions.jsonl", "w") as f:
        for row in transactions:
            f.write(json.dumps(row) + "\n")
    pd.DataFrame(
        {"client_id": list(labels), "cutoff_date": "2026-01-01", "target_next_recurring_merchant": list(labels.values())}
    ).to_csv(project.raw / "train_labels.csv", index=False)


def evaluate_on_train(project, *train_args):
    """Train rules, score it on the train split; return (log row, per-Client predictions)."""
    assert project.run("train", "--model", "rules", *train_args) == 0
    assert project.run("evaluate", "--model", "rules", "--split", "train", "--change", "E1") == 0
    row = read_rows(project.log)[-1]
    saved = pd.read_csv(project.log.parent / "runs" / f"{row['run_id']}.csv", dtype=str, keep_default_na=False)
    return row, dict(zip(saved["client_id"], saved["predicted"]))


# Cutoff 2026-01-01; monthly streams project their next payment 30 days after the last one.
TWO_STREAMS = {
    # gym is due soonest (~Jan 9); cloud is paid more recently, more often, cheaper, and sorts first.
    "T_GYM": payments("T_GYM", "gym membership", "7997", 66.0, "2025-12-10", 4)
    + payments("T_GYM", "cloud backup", "5732", 6.8, "2025-12-25", 8),
    # cloud is due soonest (~Jan 7); mobile is paid more recently, more often, pricier, and sorts last.
    "T_CLOUD": payments("T_CLOUD", "cloud storage", "5732", 6.8, "2025-12-08", 4)
    + payments("T_CLOUD", "phone contract", "4814", 47.0, "2025-12-28", 6),
}
STOPPED = {
    # an insurance stream that stopped in August, and a gym series of only two payments: neither is Active
    "T_STOPPED": payments("T_STOPPED", "insurance monthly", "6300", 109.0, "2025-08-15", 5)
    + payments("T_STOPPED", "urban gym", "7997", 50.0, "2025-12-20", 2),
}
NO_STREAMS = {"T_SHOPS": payments("T_SHOPS", "neighborhood market", "5411", 42.1, "2025-12-11", 4)}
MORE = {
    # the true family (mobile) is among the Active Streams, but software is due sooner
    "T_WRONG": payments("T_WRONG", "saas billing", "5734", 39.0, "2025-12-05", 5)
    + payments("T_WRONG", "phone contract", "4814", 47.0, "2025-12-26", 5),
    "T_NONE": payments("T_NONE", "audio streaming", "5812", 13.5, "2025-12-18", 6),
}
FIXTURE_LABELS = {
    "T_GYM": "gym",
    "T_CLOUD": "cloud",
    "T_STOPPED": "insurance",
    "T_SHOPS": "mobile",
    "T_WRONG": "mobile",
    "T_NONE": "none",
    "T_NO_TX": "none",  # a labelled Client without a single transaction
}


def all_transactions(*groups):
    return [row for group in groups for rows in group.values() for row in rows]


@pytest.fixture
def e1_train(fetched):
    write_train(fetched, all_transactions(TWO_STREAMS, STOPPED, NO_STREAMS, MORE), FIXTURE_LABELS)
    return fetched


def test_rules_picks_the_active_stream_due_soonest(e1_train):
    _, predicted = evaluate_on_train(e1_train)
    assert predicted["T_GYM"] == "gym"
    assert predicted["T_CLOUD"] == "cloud"
    assert predicted["T_WRONG"] == "software"


def test_rules_predicts_none_when_every_stream_has_stopped_or_is_too_short(e1_train):
    _, predicted = evaluate_on_train(e1_train)
    assert predicted["T_STOPPED"] == "none"


def test_rules_predicts_for_every_client_including_those_without_streams(e1_train):
    _, predicted = evaluate_on_train(e1_train)
    assert sorted(predicted) == sorted(FIXTURE_LABELS)
    assert predicted["T_SHOPS"] == "none"
    assert predicted["T_NO_TX"] == "none"
    assert predicted["T_NONE"] == "music"  # the rule cannot know this stream will not recur


def test_rules_ordering_is_selectable(e1_train):
    _, predicted = evaluate_on_train(e1_train, "--ordering", "most_recent")
    assert (predicted["T_GYM"], predicted["T_CLOUD"], predicted["T_WRONG"]) == ("cloud", "mobile", "mobile")
    _, predicted = evaluate_on_train(e1_train, "--ordering", "longest")
    assert (predicted["T_GYM"], predicted["T_CLOUD"]) == ("cloud", "mobile")
    _, predicted = evaluate_on_train(e1_train, "--ordering", "soonest")
    assert (predicted["T_GYM"], predicted["T_CLOUD"], predicted["T_WRONG"]) == ("gym", "cloud", "software")


def test_rules_log_row_records_coverage_and_selection_accuracy(e1_train, capsys):
    # Five Clients carry a family label. The true family is among the surviving (Active, due within the
    # Horizon) streams for T_GYM, T_CLOUD and T_WRONG, not for T_STOPPED (stopped) or T_SHOPS (no stream):
    # coverage 3/5. Of those three, the rule picks the true family for T_GYM and T_CLOUD: 2/3.
    row, _ = evaluate_on_train(e1_train)
    assert float(row["coverage"]) == pytest.approx(3 / 5, abs=1e-4)
    assert float(row["selection_accuracy"]) == pytest.approx(2 / 3, abs=1e-4)
    out = capsys.readouterr().out
    assert "coverage 0.6000" in out and "selection accuracy 0.6667" in out


def test_rules_is_scored_like_any_model(e1_train):
    # truth:     gym cloud insurance mobile mobile   none  none
    # predicted: gym cloud none      none   software music none
    # gym 1, cloud 1, none 2*(1/3)(1/2)/(5/6) = 0.4, the rest 0.
    row, _ = evaluate_on_train(e1_train)
    assert float(row["f1_gym"]) == pytest.approx(1.0)
    assert float(row["f1_cloud"]) == pytest.approx(1.0)
    assert float(row["f1_none"]) == pytest.approx(0.4, abs=1e-4)
    assert float(row["macro_f1"]) == pytest.approx(2.4 / 8, abs=1e-4)


def test_prior_rows_leave_the_rules_diagnostics_blank(fetched):
    assert fetched.run("train", "--model", "prior") == 0
    assert fetched.run("evaluate", "--model", "prior") == 0
    row = read_rows(fetched.log)[-1]
    assert row["coverage"] == "" and row["selection_accuracy"] == ""


# --- only Active Streams due within the Horizon -------------------------------


# With a 120-day period the projected next payment can fall beyond the 90-day Horizon.
QUARTERLY = {
    # the only Active Stream is due 2026-04-14, 103 days after the Cutoff
    "H_LATE": payments("H_LATE", "gym membership", "7997", 66.0, "2025-12-15", 3, every_days=120),
    # gym is paid most recently but due beyond the Horizon; cloud is due 2026-02-17
    "H_TWO": payments("H_TWO", "gym membership", "7997", 66.0, "2025-12-15", 3, every_days=120)
    + payments("H_TWO", "cloud backup", "5732", 6.8, "2025-10-20", 3, every_days=120),
    "H_IN": payments("H_IN", "cloud backup", "5732", 6.8, "2025-10-20", 3, every_days=120),
}


def test_rules_considers_only_streams_due_within_the_horizon(fetched):
    write_train(fetched, all_transactions(QUARTERLY), {"H_LATE": "gym", "H_TWO": "gym", "H_IN": "cloud"})
    row, predicted = evaluate_on_train(
        fetched, "--ordering", "most_recent", "--param", "canonical_periods=120"
    )
    assert predicted == {"H_LATE": "none", "H_TWO": "cloud", "H_IN": "cloud"}
    # gym is not a surviving stream for either gym-labelled Client
    assert float(row["coverage"]) == pytest.approx(1 / 3, abs=1e-4)
    assert float(row["selection_accuracy"]) == pytest.approx(1.0)


def test_rules_stream_parameters_are_kept_with_the_trained_model(fetched):
    write_train(fetched, all_transactions(QUARTERLY), {"H_LATE": "gym", "H_TWO": "gym", "H_IN": "cloud"})
    # min_payments=4: no stream here has four payments, so none is Active
    _, predicted = evaluate_on_train(fetched, "--param", "min_payments=4")
    assert set(predicted.values()) == {"none"}


# --- every split, every Client --------------------------------------------------


def test_rules_on_the_fixture_selection_set_and_test_split(fetched):
    # selection set: C000011 (coffee shop only, none), C000013 (gym), C000014 (cloud); all correct.
    assert fetched.run("train", "--model", "rules") == 0
    out = fetched.root / "proba.csv"
    assert fetched.run("evaluate", "--model", "rules", "--proba", str(out)) == 0
    row = read_rows(fetched.log)[-1]
    assert float(row["macro_f1"]) == pytest.approx(3 / 8, abs=1e-4)
    proba = pd.read_csv(out).set_index("client_id")
    assert list(proba.columns) == LABELS
    assert (proba.sum(axis=1) == 1.0).all()
    assert proba.idxmax(axis=1).to_dict() == {"C000011": "none", "C000013": "gym", "C000014": "cloud"}

    assert fetched.run("submit", "--model", "rules", "--name", "e1") == 0
    sub = pd.read_csv(fetched.root / "submissions" / "e1.csv", dtype=str)
    assert dict(zip(sub["client_id"], sub["predicted_next_recurring_merchant"])) == {
        "C000004": "streaming", "C000008": "mobile", "C000010": "none",
    }


@pytest.mark.parametrize("option", [["--ordering", "soonest"], ["--param", "min_payments=4"]])
def test_rules_options_are_refused_for_other_models(fetched, option):
    with pytest.raises(SystemExit):
        fetched.run("train", "--model", "prior", *option)
    assert not (fetched.root / "artifacts" / "prior.json").exists()


def test_unknown_ordering_is_refused(fetched):
    with pytest.raises(SystemExit):
        fetched.run("train", "--model", "rules", "--ordering", "alphabetical")
