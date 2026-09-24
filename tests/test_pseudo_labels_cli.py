"""Seam 1: `rf pseudo-labels` and its fidelity check through the CLI against the fixture data."""

import csv
import json

import pandas as pd
import pytest

from recurring_family import data
from recurring_family import streams as stream_detection

LABELS = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none"]

# train fixture, monthly payments 2025-09-05 .. 2025-12-05: at the default Shifted Cutoff (2025-10-03)
# C000001 gym, C000002 cloud and C000007 insurance pay first on 2025-10-05, inside the Horizon, in
# four-payment streams; the others only shop.
TRAIN_PSEUDO = {
    "C000001": "gym", "C000002": "cloud", "C000003": "none",
    "C000005": "none", "C000007": "insurance", "C000009": "none",
}


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run(project, capsys, *args):
    """Run `pseudo-labels`; return (exit code, stdout, stderr)."""
    capsys.readouterr()
    code = project.run("pseudo-labels", *args)
    out = capsys.readouterr()
    return code, out.out, out.err


def distribution(out):
    """The printed per-label lines: label -> (count, share)."""
    rows = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] in LABELS:
            rows[parts[0]] = (int(parts[1]), float(parts[2]))
    return rows


def source(out):
    header = out.splitlines()[0]
    return "cached" if "[cached]" in header else "labelled" if "[labelled]" in header else header


# --- the table ------------------------------------------------------------------------


def test_writes_one_row_per_client_and_prints_the_label_distribution(fetched, capsys):
    out_csv = fetched.root / "pl.csv"
    code, out, _ = run(fetched, capsys, "--split", "train", "--cutoff", "2025-10-03", "--min-payments", "2",
                       "--out", str(out_csv))
    assert code == 0
    rows = read_rows(out_csv)
    assert list(rows[0]) == ["client_id", "cutoff_date", "target_next_recurring_merchant"]
    assert {r["client_id"]: r["target_next_recurring_merchant"] for r in rows} == TRAIN_PSEUDO
    assert {r["cutoff_date"] for r in rows} == {"2025-10-03"}
    assert len(rows) == len({r["client_id"] for r in rows})

    printed = distribution(out)
    assert list(printed) == LABELS  # every allowed label, in the fixed order, even at 0
    assert printed["none"] == (3, pytest.approx(0.5))
    for family in ("gym", "cloud", "insurance"):
        assert printed[family] == (1, pytest.approx(1 / 6, abs=1e-4))
    for family in ("mobile", "music", "software", "streaming"):
        assert printed[family] == (0, 0.0)
    assert "6 Clients" in out and "2025-10-03" in out and str(out_csv) in out


def test_the_default_table_goes_under_artifacts_and_names_its_parameters(fetched, capsys):
    code, out, _ = run(fetched, capsys, "--split", "train", "--min-payments", "5")
    assert code == 0
    path = fetched.root / "artifacts" / "pseudo_labels" / "train-2025-10-03-min5.csv"
    assert str(path) in out
    # four-payment streams are too short at a five-payment minimum
    assert {r["target_next_recurring_merchant"] for r in read_rows(path)} == {"none"}


def test_the_default_cutoff_is_the_latest_fully_observed_shifted_cutoff(fetched, capsys):
    a = fetched.root / "a.csv"
    b = fetched.root / "b.csv"
    assert run(fetched, capsys, "--split", "train", "--min-payments", "2", "--out", str(a))[0] == 0
    assert run(fetched, capsys, "--split", "train", "--cutoff", "2025-10-03", "--min-payments", "2",
               "--out", str(b))[0] == 0
    assert a.read_bytes() == b.read_bytes()


def test_an_earlier_shifted_cutoff_moves_the_horizon(fetched, capsys):
    out_csv = fetched.root / "pl.csv"
    # from 2025-09-01 the Horizon's first payments are the 2025-09-05 ones: same families
    assert run(fetched, capsys, "--split", "train", "--cutoff", "2025-09-01", "--min-payments", "2",
               "--out", str(out_csv))[0] == 0
    rows = read_rows(out_csv)
    assert {r["client_id"]: r["target_next_recurring_merchant"] for r in rows} == TRAIN_PSEUDO
    assert {r["cutoff_date"] for r in rows} == {"2025-09-01"}


@pytest.mark.parametrize("cutoff", ["2025-10-04", "2026-01-01"])
def test_a_shifted_cutoff_whose_horizon_is_not_fully_observed_is_refused(fetched, capsys, cutoff):
    code, _, err = run(fetched, capsys, "--split", "train", "--cutoff", cutoff)
    assert code != 0
    assert "2025-12-31" in err and "latest Shifted Cutoff is 2025-10-03" in err
    assert not list((fetched.root / "artifacts").rglob("*.csv"))


# --- no label file is read ------------------------------------------------------------


@pytest.mark.parametrize("split", ["valid", "test", "unlabeled", "train"])
def test_pseudo_labels_for_any_split_read_no_label_file(fetched, capsys, monkeypatch, split):
    # no label file exists, and any attempt to read one fails
    for path in fetched.raw.glob("*_labels.csv"):
        path.unlink()

    def refuse(*args, **kwargs):
        raise AssertionError("a label file was read")

    monkeypatch.setattr(data, "_read_label_file", refuse)
    out_csv = fetched.root / f"{split}.csv"
    code, out, err = run(fetched, capsys, "--split", split, "--min-payments", "2", "--out", str(out_csv))
    assert code == 0, err
    clients = data.load_transactions(fetched.raw, split)["client_id"].unique()
    rows = read_rows(out_csv)
    assert sorted(r["client_id"] for r in rows) == sorted(clients)
    assert set(r["target_next_recurring_merchant"] for r in rows) <= set(LABELS)
    assert sum(n for n, _ in distribution(out).values()) == len(clients)


@pytest.mark.parametrize("split", ["valid", "test"])
def test_pseudo_labels_on_valid_and_test_trip_no_label_guard(fetched, capsys, split):
    # the label files are present, the guards are live: the command neither reads nor is refused
    code, _, err = run(fetched, capsys, "--split", split, "--min-payments", "2")
    assert code == 0 and "labels" not in err


def test_a_label_read_during_pseudo_labelling_fails_loudly(fetched):
    with data.pseudo_labelling():
        for source in ("train", "valid", "selection", "holdout"):
            with pytest.raises(data.LabelLeak, match="Pseudo-Labels come from transactions only"):
                data.load_labels(fetched.raw, source)


# --- the cache ------------------------------------------------------------------------


def test_a_second_run_with_the_same_parameters_loads_from_the_cache(fetched, capsys):
    cache = fetched.root / "artifacts" / "pseudo_labels" / "cache"
    code, out, _ = run(fetched, capsys, "--split", "train", "--min-payments", "2")
    assert (code, source(out)) == (0, "labelled")
    stamps = {f: f.stat().st_mtime_ns for f in cache.iterdir()}
    assert len(stamps) == 1

    code, again, _ = run(fetched, capsys, "--split", "train", "--min-payments", "2")
    assert (code, source(again)) == (0, "cached")
    assert {f: f.stat().st_mtime_ns for f in cache.iterdir()} == stamps
    assert distribution(again) == distribution(out)


@pytest.mark.parametrize(
    "changed",
    [
        ["--split", "valid"],
        ["--cutoff", "2025-09-01"],
        ["--min-payments", "3"],
        ["--param", "amount_tolerance=0.2"],
        ["--param", "refund_window_days=3"],
        ["--min-payments-before", "1"],
        ["--churn", "0.2"],
    ],
    ids=[
        "split", "cutoff", "min-payments", "stream-param", "another-stream-param", "min-payments-before", "churn",
    ],
)
def test_changing_any_parameter_misses_the_cache(fetched, capsys, changed):
    base = {"--split": "train", "--cutoff": "2025-10-03", "--min-payments": "2"}
    args = [x for kv in base.items() for x in kv]
    assert source(run(fetched, capsys, *args)[1]) == "labelled"
    assert source(run(fetched, capsys, *args)[1]) == "cached"
    merged = dict(base)
    extra = []
    if changed[0] not in merged:
        extra = changed
    else:
        merged[changed[0]] = changed[1]
    other = [x for kv in merged.items() for x in kv] + extra
    assert source(run(fetched, capsys, *other)[1]) == "labelled"
    assert source(run(fetched, capsys, *other)[1]) == "cached"
    assert source(run(fetched, capsys, *args)[1]) == "cached"  # the first table is still there


def test_a_changed_detector_misses_the_cache(fetched, capsys, monkeypatch):
    args = ("--split", "train", "--min-payments", "2")
    assert source(run(fetched, capsys, *args)[1]) == "labelled"
    monkeypatch.setattr(stream_detection, "_detector_version", lambda: "an edited detector")
    code, out, _ = run(fetched, capsys, *args)
    assert (code, source(out)) == (0, "labelled")
    assert source(run(fetched, capsys, *args)[1]) == "cached"


def test_a_changed_family_table_relabels(fetched, capsys, monkeypatch):
    args = ("--split", "train", "--min-payments", "2")
    assert distribution(run(fetched, capsys, *args)[1])["gym"][0] == 1
    monkeypatch.delitem(stream_detection.HOME_MCC, "7997")  # gym payments lose their home MCC
    code, out, _ = run(fetched, capsys, *args)
    assert (source(out), distribution(out)["gym"][0]) == ("labelled", 0)


# --- the fidelity check ---------------------------------------------------------------


def _tx(client, day, amount, description, mcc):
    return {
        "amount": amount, "client_id": client, "currency": "eur", "description": description,
        "direction": "out", "fee": 0.0, "mcc": mcc, "timestamp": f"{day}T10:00:00Z", "type": "card_payment",
    }


def _monthly(client, description, mcc, amount, months):
    return [_tx(client, f"2025-{m:02d}-{day:02d}", amount, description, mcc) for m, day in months]


def fidelity_train(project):
    """A train split built for the check, with the Shifted Cutoff 2025-10-03:

    - A: gym every month 2025-01-05 .. 2025-12-05. Real label gym; the rule says gym at both Cutoffs.
    - B: cloud starts inside the Horizon (10-10, 11-10, 12-10). Real label none (the rule agrees: three
      payments are gated). Pseudo-Label cloud when two or three payments suffice, else none.
    - C: insurance every month to 2025-09-05, then stops. Real label none; at the Shifted Cutoff it
      is an Active Stream due 10-05, so the rule says insurance, but its Pseudo-Label is none. Only
      a rule that saw the Horizon's transactions would know.
    - D: coffee shop only. None everywhere.
    """
    rows = _monthly("A", "gym membership", "7997", 66.0, [(m, 5) for m in range(1, 13)])
    rows += _monthly("B", "cloud backup", "5732", 6.8, [(10, 10), (11, 10), (12, 10)])
    rows += _monthly("C", "cover plan", "6300", 109.0, [(m, 5) for m in range(1, 10)])
    rows += _monthly("D", "coffee shop", "5812", 4.0, [(m, 7) for m in range(1, 13)])
    with open(project.raw / "train_transactions.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    labels = pd.DataFrame(
        {"client_id": list("ABCD"), "cutoff_date": "2026-01-01",
         "target_next_recurring_merchant": ["gym", "none", "none", "none"]}
    )
    labels.to_csv(project.raw / "train_labels.csv", index=False)
    return project


# Real: truth gym/none/none/none, rule gym/none/none/none -> F1 gym 1, none 1: macro 2/8 = 0.25; none share 0.75.
# Shifted rule: gym, none, insurance, none.
# min_payments 2 or 3: Pseudo-Labels gym, cloud, none, none (none share 0.5, gap -0.25);
#   F1 gym 1, cloud 0, insurance 0, none 0.5: macro 1.5/8 = 0.1875 (gap -0.0625).
# min_payments 4 and up: gym, none, none, none (none share 0.75, gap 0);
#   F1 gym 1, insurance 0, none 0.8 (tp 2, precision 1, recall 2/3): macro 1.8/8 = 0.225 (gap -0.025).
REAL_F1 = 0.25


def fidelity_log_rows(project):
    return [r for r in read_rows(project.log) if r["row_type"] == "fidelity"]


def test_the_fidelity_check_chooses_min_payments_and_passes(fetched, capsys):
    fidelity_train(fetched)
    code, out, err = run(fetched, capsys, "--split", "train", "--fidelity")
    assert code == 0, err
    report = out[out.index("fidelity"):]
    # both comparisons, each with its tolerance
    assert "none share 0.7500 vs real 0.7500: gap +0.0000, tolerance 0.05" in report
    assert "rule macro-F1 0.2250 vs real 0.2500: gap -0.0250, tolerance 0.05" in report
    assert "chosen min_payments 4" in report
    assert report.rstrip().endswith("PASS")

    # the chosen setting's table is written
    table = fetched.root / "artifacts" / "pseudo_labels" / "train-2025-10-03-min4.csv"
    assert {r["client_id"]: r["target_next_recurring_merchant"] for r in read_rows(table)} == {
        "A": "gym", "B": "none", "C": "none", "D": "none"
    }

    [row] = fidelity_log_rows(fetched)
    assert (row["split"], row["model"], row["verdict"]) == ("train", "rules+gate4", "pass")
    assert float(row["macro_f1"]) == pytest.approx(0.225, abs=1e-4)
    assert float(row["f1_gym"]) == pytest.approx(1.0) and float(row["f1_none"]) == pytest.approx(0.8)
    assert float(row["delta"]) == pytest.approx(0.225 - REAL_F1, abs=1e-4)
    for text in ("2025-10-03", "min_payments 4", "2-10", "90"):
        assert text in row["change"], text
    for text in ("0.7500", "0.2500", "0.05"):
        assert text in row["conclusion"], text
    assert row["run_id"]


MIN_PAYMENTS_ONLY = ("--before-candidates", "0", "--churn-candidates", "0")


def test_the_fidelity_check_fails_when_no_setting_is_within_tolerance(fetched, capsys):
    fidelity_train(fetched)
    code, out, _ = run(fetched, capsys, "--split", "train", "--fidelity", "--candidates", "2,3", *MIN_PAYMENTS_ONLY)
    assert code == 0  # a failed check is a result, not an error
    report = out[out.index("fidelity"):]
    assert "none share 0.5000 vs real 0.7500: gap -0.2500, tolerance 0.05" in report
    assert "rule macro-F1 0.1875 vs real 0.2500: gap -0.0625, tolerance 0.05" in report
    assert "chosen min_payments 2, min_payments_before 0, churn 0" in report
    assert report.rstrip().endswith("FAIL")
    [row] = fidelity_log_rows(fetched)
    assert row["verdict"] == "fail" and "min_payments 2" in row["change"] and "2-3" in row["change"]


def test_the_fidelity_check_prints_every_setting_it_chose_among(fetched, capsys):
    fidelity_train(fetched)
    _, out, _ = run(
        fetched, capsys, "--split", "train", "--fidelity", "--candidates", "2,5", "--before-candidates", "0,1",
        "--churn-candidates", "0,0.5",
    )
    table = {tuple(line.split()[:3]): line.split()[3:] for line in out.splitlines() if line.strip()[:1].isdigit()}
    assert len(table) == 2 * 2 * 2
    assert table[("2", "0", "0")][:2] == ["0.5000", "0.1875"] and table[("5", "0", "0")][:2] == ["0.7500", "0.2250"]
    # B's cloud stream starts inside the Horizon: with a payment needed before the Shifted Cutoff it is none
    assert table[("2", "1", "0")][:2] == ["0.7500", "0.2250"]
    [row] = fidelity_log_rows(fetched)
    for text in ("min_payments 2,5", "min_payments_before 0-1", "churn 0,0.5", "8 settings"):
        assert text in row["change"], text


def test_the_fidelity_check_skips_settings_that_repeat_another(fetched, capsys):
    # two payments before the Shifted Cutoff and one in the Horizon make three: a minimum of two in all
    # never binds then, so (min_payments 2, before 2) would repeat (3, 2)
    fidelity_train(fetched)
    _, out, _ = run(
        fetched, capsys, "--split", "train", "--fidelity", "--candidates", "2,3", "--before-candidates", "0,2",
        "--churn-candidates", "0",
    )
    table = [tuple(line.split()[:3]) for line in out.splitlines() if line.strip()[:1].isdigit()]
    assert table == [("2", "0", "0"), ("3", "0", "0"), ("3", "2", "0")]
    [row] = fidelity_log_rows(fetched)
    assert "3 settings" in row["change"]


def test_a_minimum_before_above_every_minimum_in_all_is_still_searched(fetched, capsys):
    # three payments before the Shifted Cutoff need four in all: searched as min_payments 4, not dropped
    fidelity_train(fetched)
    _, out, _ = run(
        fetched, capsys, "--split", "train", "--fidelity", "--candidates", "2,3", "--before-candidates", "3",
        "--churn-candidates", "0",
    )
    table = [tuple(line.split()[:3]) for line in out.splitlines() if line.strip()[:1].isdigit()]
    assert table == [("4", "3", "0")]


def test_the_fidelity_check_keeps_every_settings_verdict_next_to_the_log(fetched, capsys):
    fidelity_train(fetched)
    run(fetched, capsys, "--split", "train", "--fidelity", "--candidates", "2,5", "--churn-candidates", "0,0.5")
    [row] = fidelity_log_rows(fetched)
    kept = read_rows(fetched.root / "experiments" / "fidelity" / f"{row['run_id']}.csv")
    assert len(kept) == 2 * 4 * 2  # min_payments 2 with 2 or 3 payments before is searched as 3 or 4
    assert {r["cutoff"] for r in kept} == {"2025-10-03"}
    by_setting = {(r["min_payments"], r["min_payments_before"], r["churn"]): r for r in kept}
    assert by_setting[("5", "0", "0")]["verdict"] == "pass" and by_setting[("5", "0", "0")]["none_share"] == "0.7500"
    assert by_setting[("2", "0", "0")]["verdict"] == "fail" and by_setting[("2", "0", "0")]["rule_gap"] == "-0.0625"


def test_the_fidelity_check_leaves_other_log_rows_and_best_runs_alone(fetched, capsys):
    fidelity_train(fetched)
    assert run(fetched, capsys, "--split", "train", "--fidelity")[0] == 0
    assert run(fetched, capsys, "--split", "train", "--fidelity", "--candidates", "2", *MIN_PAYMENTS_ONLY)[0] == 0
    rows = read_rows(fetched.log)
    assert [r["verdict"] for r in rows] == ["pass", "fail"]
    assert {r["row_type"] for r in rows} == {"fidelity"}


@pytest.mark.parametrize("split", ["valid", "test", "unlabeled"])
def test_the_fidelity_check_runs_on_train_only(fetched, capsys, split):
    code, _, err = run(fetched, capsys, "--split", split, "--fidelity")
    assert code != 0 and "train" in err
    assert not fetched.log.exists()


@pytest.mark.parametrize("setting", [("--min-payments", "3"), ("--min-payments-before", "1"), ("--churn", "0.2")])
def test_the_fidelity_check_chooses_the_labeller_settings_itself(fetched, capsys, setting):
    with pytest.raises(SystemExit):
        fetched.run("pseudo-labels", "--split", "train", "--fidelity", *setting)
    assert "--candidates" in capsys.readouterr().err
    assert not fetched.log.exists()


@pytest.mark.parametrize(
    "args",
    [
        ("--candidates", "2,3"),
        ("--before-candidates", "0,1"),
        ("--churn-candidates", "0,0.2"),
        ("--churn", "1"),
        ("--churn", "-0.1"),
        ("--min-payments-before", "-1"),
    ],
)
def test_bad_labeller_options_are_refused(fetched, args):
    with pytest.raises(SystemExit) as e:
        fetched.run("pseudo-labels", "--split", "train", *args)
    assert e.value.code != 0


# --- labeller settings ----------------------------------------------------------------


def test_the_labeller_settings_reach_the_table_and_its_name(fetched, capsys):
    # B's cloud stream starts inside the Horizon: with a payment needed before the Shifted Cutoff it is none
    fidelity_train(fetched)
    code, out, _ = run(fetched, capsys, "--split", "train", "--min-payments", "2", "--min-payments-before", "1")
    assert code == 0
    table = fetched.root / "artifacts" / "pseudo_labels" / "train-2025-10-03-min2-before1.csv"
    assert "min_payments 2, min_payments_before 1, churn 0" in out and str(table) in out
    assert {r["client_id"]: r["target_next_recurring_merchant"] for r in read_rows(table)} == {
        "A": "gym", "B": "none", "C": "none", "D": "none"
    }

    code, out, _ = run(fetched, capsys, "--split", "train", "--min-payments", "2", "--churn", "0.999")
    assert code == 0
    table = fetched.root / "artifacts" / "pseudo_labels" / "train-2025-10-03-min2-churn0.999.csv"
    assert {r["target_next_recurring_merchant"] for r in read_rows(table)} == {"none"}  # every Client churns
    assert distribution(out)["none"] == (4, 1.0)


def churn_train(project, n=40):
    """n Clients who all pay a gym membership every month of 2025, half of them labelled gym and half
    none: the real task's churn at the Cutoff, which no stream inside the known history shows. So at the
    Shifted Cutoff every Pseudo-Label is gym unless the labeller churns Clients."""
    rows, labels = [], {}
    for i in range(n):
        client = f"G{i:03d}"
        rows += _monthly(client, "gym membership", "7997", 66.0, [(m, 5) for m in range(1, 13)])
        labels[client] = "gym" if i % 2 else "none"
    with open(project.raw / "train_transactions.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    pd.DataFrame(
        {"client_id": list(labels), "cutoff_date": "2026-01-01", "target_next_recurring_merchant": list(labels.values())}
    ).to_csv(project.raw / "train_labels.csv", index=False)


def test_the_fidelity_check_chooses_churn_when_that_closes_the_gaps(fetched, capsys):
    churn_train(fetched)
    code, out, _ = run(
        fetched, capsys, "--split", "train", "--fidelity", "--candidates", "2", "--before-candidates", "0",
        "--churn-candidates", "0,0.5",
    )
    assert code == 0
    report = out[out.index("fidelity"):]
    # without churn every Pseudo-Label is gym: none share 0 against a real 0.5
    assert "  2             0       0      0.0000" in report
    assert "chosen min_payments 2, min_payments_before 0, churn 0.5" in report
    assert report.rstrip().endswith("PASS")
    table = fetched.root / "artifacts" / "pseudo_labels" / "train-2025-10-03-min2-churn0.5.csv"
    churned = [r["target_next_recurring_merchant"] for r in read_rows(table)]
    assert set(churned) == {"gym", "none"}
    assert f"none share {churned.count('none') / len(churned):.4f} vs real 0.5000" in report
    [row] = fidelity_log_rows(fetched)
    assert row["verdict"] == "pass" and "churn 0.5" in row["change"]
