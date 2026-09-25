"""Seam 1: the pipeline through the CLI against the fixture data folder."""

import argparse
import csv
import dataclasses
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import FIXTURE_DATA
from recurring_family import streams as stream_detection
from recurring_family.cli import stream_param
from recurring_family.streams import StreamParams

EXPECTED_FILES = sorted(p.name for p in FIXTURE_DATA.iterdir())
LABELS = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none"]
TEST_IDS = ["C000004", "C000008", "C000010"]


def snapshot(folder):
    return {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(folder.iterdir())}


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --- fetch-data ---------------------------------------------------------------


def test_fetch_data_unpacks_every_file_into_raw_area(project):
    assert project.run("fetch-data", "--zip", str(project.zip)) == 0
    assert sorted(p.name for p in project.raw.iterdir()) == EXPECTED_FILES
    for name in EXPECTED_FILES:
        assert (project.raw / name).read_bytes() == (FIXTURE_DATA / name).read_bytes()


def test_fetch_data_twice_changes_nothing(fetched):
    before = snapshot(fetched.raw)
    assert fetched.run("fetch-data", "--zip", str(fetched.zip)) == 0
    assert snapshot(fetched.raw) == before


def test_fetch_data_repairs_a_corrupted_file(fetched):
    target = fetched.raw / "train_labels.csv"
    target.write_text("garbage\n")
    assert fetched.run("fetch-data", "--zip", str(fetched.zip)) == 0
    assert target.read_bytes() == (FIXTURE_DATA / "train_labels.csv").read_bytes()


# --- train + evaluate ---------------------------------------------------------


def test_prior_model_on_fixture_scores_hand_computed_macro_f1(fetched):
    # evaluate scores the selection set by default: valid minus the sealed
    # holdout (C000012), so truth is none, gym, cloud. The prior (train
    # majority = none) predicts all none: F1(none) = 2 * 1/3 * 1 / (4/3) = 1/2.
    assert fetched.run("train", "--model", "prior") == 0
    assert fetched.run("evaluate", "--model", "prior", "--change", "prior baseline") == 0

    rows = read_csv_rows(fetched.log)
    assert len(rows) == 1
    row = rows[0]
    assert float(row["macro_f1"]) == pytest.approx((1 / 2) / 8, abs=1e-4)
    assert float(row["f1_none"]) == pytest.approx(1 / 2, abs=1e-4)
    for family in LABELS[:-1]:
        # gym and cloud are in the truth but never predicted: they must be 0,
        # and absent families still appear in the log (fixed eight-label set).
        assert float(row[f"f1_{family}"]) == 0.0


def test_evaluate_appends_one_log_row_per_run(fetched):
    assert fetched.run("train", "--model", "prior") == 0
    assert fetched.run("evaluate", "--model", "prior", "--change", "first") == 0
    assert fetched.run(
        "evaluate", "--model", "prior", "--change", "second", "--conclusion", "same as first"
    ) == 0

    rows = read_csv_rows(fetched.log)
    assert [r["change"] for r in rows] == ["first", "second"]
    assert rows[1]["conclusion"] == "same as first"
    assert set(rows[0]) >= {"date", "change", "macro_f1", "conclusion"} | {f"f1_{l}" for l in LABELS}
    assert rows[0]["date"]


def test_evaluate_without_training_fails(fetched):
    assert fetched.run("evaluate", "--model", "prior") != 0
    assert not fetched.log.exists()


def test_prior_model_learns_class_frequencies_from_train_labels(fetched):
    # Flip train labels so gym is the majority: the prior must follow the labels.
    labels = pd.read_csv(fetched.raw / "train_labels.csv")
    labels["target_next_recurring_merchant"] = ["gym", "gym", "gym", "none", "cloud", "none"]
    labels.to_csv(fetched.raw / "train_labels.csv", index=False)

    assert fetched.run("train", "--model", "prior") == 0
    assert fetched.run("evaluate", "--model", "prior") == 0
    row = read_csv_rows(fetched.log)[0]
    # selection truth has one gym among three: precision 1/3, recall 1 -> F1 0.5
    assert float(row["f1_gym"]) == pytest.approx(0.5, abs=1e-4)
    assert float(row["f1_none"]) == 0.0

    assert fetched.run("submit", "--model", "prior", "--name", "gym") == 0
    sub = pd.read_csv(fetched.root / "submissions" / "gym.csv")
    assert set(sub["predicted_next_recurring_merchant"]) == {"gym"}


def test_prior_model_returns_train_class_frequencies_for_every_client(fetched):
    # train labels: gym, cloud, none, none, insurance, none
    # -> none 3/6, gym/cloud/insurance 1/6 each, other families 0.
    assert fetched.run("train", "--model", "prior") == 0
    out = fetched.root / "proba.csv"
    assert fetched.run("evaluate", "--model", "prior", "--proba", str(out)) == 0

    proba = pd.read_csv(out).set_index("client_id")
    assert list(proba.columns) == LABELS
    assert proba.index.tolist() == ["C000011", "C000013", "C000014"]  # the selection set
    expected = {"none": 0.5, "gym": 1 / 6, "cloud": 1 / 6, "insurance": 1 / 6,
                "mobile": 0.0, "music": 0.0, "software": 0.0, "streaming": 0.0}
    for cid, row in proba.iterrows():
        assert row.sum() == pytest.approx(1.0, abs=1e-9)
        for label, value in expected.items():
            assert row[label] == pytest.approx(value, abs=1e-9), (cid, label)


# --- submit -------------------------------------------------------------------


def test_prior_submission_is_the_all_none_safety_file(fetched):
    assert fetched.run("train", "--model", "prior") == 0
    assert fetched.run("submit", "--model", "prior", "--name", "all-none") == 0

    path = fetched.root / "submissions" / "all-none.csv"
    sub = pd.read_csv(path)
    assert list(sub.columns) == ["client_id", "predicted_next_recurring_merchant"]
    assert sorted(sub["client_id"]) == TEST_IDS
    assert set(sub["predicted_next_recurring_merchant"]) == {"none"}
    assert fetched.run("submit", "--check", str(path)) == 0


def write_submission(project, rows, name="candidate.csv"):
    path = project.root / name
    with open(path, "w", newline="") as f:
        f.write("client_id,predicted_next_recurring_merchant\n")
        for cid, label in rows:
            f.write(f"{cid},{label}\n")
    return path


def test_submit_check_accepts_a_valid_file(fetched):
    rows = list(zip(TEST_IDS, ["gym", "none", "streaming"]))
    assert fetched.run("submit", "--check", str(write_submission(fetched, rows))) == 0


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([("C000004", "none"), ("C000008", "none")], id="missing"),
        pytest.param([(c, "none") for c in TEST_IDS] + [("C999999", "none")], id="extra"),
        pytest.param([(c, "none") for c in TEST_IDS] + [("C000004", "none")], id="duplicate"),
        pytest.param([("C000004", "netflix"), ("C000008", "none"), ("C000010", "none")], id="bad-label"),
        pytest.param([("C000004", ""), ("C000008", "none"), ("C000010", "none")], id="empty-label"),
    ],
)
def test_submit_check_rejects_invalid_files(fetched, rows, capsys):
    path = write_submission(fetched, rows)
    assert fetched.run("submit", "--check", str(path)) != 0
    assert "invalid" in capsys.readouterr().err.lower()


def test_submit_check_rejects_wrong_column_names(fetched):
    path = fetched.root / "bad.csv"
    path.write_text("client_id,label\n" + "".join(f"{c},none\n" for c in TEST_IDS))
    assert fetched.run("submit", "--check", str(path)) != 0


# --- fetch-data: typed load of every split ------------------------------------
# Hand-counted from tests/fixtures/data/*.jsonl: every split spans the same
# first/last timestamp; Client and transaction counts differ per split.


def refetch_with(project, tmp_path, name, edit):
    """Zip a copy of the fixture data with one file edited, then fetch-data it."""
    from conftest import make_zip

    src = tmp_path / "edited"
    src.mkdir()
    for p in FIXTURE_DATA.iterdir():
        (src / p.name).write_bytes(p.read_bytes())
    target = src / name
    target.write_text(edit(target.read_text()))
    return project.run("fetch-data", "--zip", str(make_zip(src, tmp_path / "edited.zip")))


def test_fetch_data_reports_every_split_loaded_with_utc_timestamps(project, capsys):
    assert project.run("fetch-data", "--zip", str(project.zip)) == 0
    out = capsys.readouterr().out
    span = "2025-09-05T10:00:00Z .. 2025-12-05T10:00:00Z"
    assert f"train: 6 Clients, 30 transactions, {span}" in out
    assert f"valid: 4 Clients, 20 transactions, {span}" in out
    assert f"test: 3 Clients, 15 transactions, {span}" in out
    assert f"unlabeled: 1 Clients, 5 transactions, {span}" in out


def test_fetch_data_converts_offset_timestamps_to_utc(project, tmp_path, capsys):
    # The last unlabeled transaction moved to 12:30 at +02:00, i.e. 10:30 UTC.
    edit = lambda s: s.replace('"2025-12-05T10:00:00Z"', '"2025-12-05T12:30:00+02:00"')
    assert refetch_with(project, tmp_path, "unlabeled_pretrain_transactions.jsonl", edit) == 0
    out = capsys.readouterr().out
    assert "unlabeled: 1 Clients, 5 transactions, 2025-09-05T10:00:00Z .. 2025-12-05T10:30:00Z" in out


@pytest.mark.parametrize(
    "bad", ['"2025-12-05T10:00:00"', '"not a date"', "null"], ids=["naive", "malformed", "missing"]
)
def test_fetch_data_fails_loudly_on_a_bad_timestamp(project, tmp_path, capsys, bad):
    edit = lambda s: s.replace('"2025-12-05T10:00:00Z"', bad, 1)
    assert refetch_with(project, tmp_path, "unlabeled_pretrain_transactions.jsonl", edit) == 1
    err = capsys.readouterr().err
    assert "unlabeled" in err and "timestamp" in err


# --- streams ------------------------------------------------------------------


def family_summary(out):
    rows = {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] in LABELS:
            rows[parts[0]] = {parts[i]: int(parts[i + 1]) for i in range(1, len(parts) - 1, 2)}
    return rows


def test_streams_prints_a_per_family_summary_of_the_split(fetched, capsys):
    # train fixture: C000001 gym, C000002 cloud, C000007 insurance (4 monthly payments each);
    # coffee-shop and grocery payments never form streams.
    capsys.readouterr()
    assert fetched.run("streams", "--split", "train") == 0
    out = capsys.readouterr().out
    summary = family_summary(out)
    assert sorted(summary) == [label for label in LABELS if label != "none"]
    expected = {"cloud", "gym", "insurance"}
    for family, counts in summary.items():
        n = 1 if family in expected else 0
        assert counts == {"streams": n, "active": n, "clients": n}, family
    assert "6 Clients" in out and "3 Recurring Streams" in out


def test_streams_table_is_cached_per_split(fetched, capsys):
    cache = fetched.root / "artifacts" / "streams"
    assert fetched.run("streams", "--split", "train") == 0
    assert fetched.run("streams", "--split", "valid") == 0
    files = sorted(cache.iterdir())
    assert [f.name.split("-")[0] for f in files] == ["train", "valid"]
    stamps = {f: f.stat().st_mtime_ns for f in files}
    capsys.readouterr()

    assert fetched.run("streams", "--split", "train") == 0
    assert "cached" in capsys.readouterr().out
    assert {f: f.stat().st_mtime_ns for f in sorted(cache.iterdir())} == stamps

    # a changed raw file invalidates the cache instead of serving a stale table
    raw = fetched.raw / "train_transactions.jsonl"
    raw.write_text("\n".join(line for line in raw.read_text().splitlines() if "7997" not in line) + "\n")
    assert fetched.run("streams", "--split", "train") == 0
    assert family_summary(capsys.readouterr().out)["gym"]["streams"] == 0


def streams_run(project, capsys, *args, split="train"):
    """Run `streams` and return (source, per-family summary): source is 'cached' or 'detected'."""
    capsys.readouterr()
    assert project.run("streams", "--split", split, *args) == 0
    out = capsys.readouterr().out
    header = out.splitlines()[0]
    source = "cached" if "[cached]" in header else "detected" if "[detected]" in header else header
    return source, family_summary(out)


# a changed value for every stream parameter; the gym fixture stream (4 payments, last 2025-12-05) goes
# inactive under the first two
CHANGED_PARAMS = {
    "min_payments": "5",
    "active_periods": "0.5",
    "amount_tolerance": "0.2",
    "canonical_periods": "7,30",
    "music_streaming_split": "20",
    "refund_window_days": "3",
    "schedule_tolerance_days": "1",
    "stray_min_period_days": "20",
    "join_strays": "false",
    "join_decoys": "true",
    "robust": "true",
}


@pytest.mark.parametrize(
    "text, value",
    [("join_strays=false", False), ("join_strays=False", False), ("join_strays=0", False), ("join_strays=no", False),
     ("join_strays=true", True), ("join_strays=1", True), ("join_strays=yes", True)],
)
def test_a_boolean_stream_parameter_parses_true_and_false_words(text, value):
    assert stream_param(text) == ("join_strays", value)


def test_a_boolean_stream_parameter_refuses_other_words():
    with pytest.raises(argparse.ArgumentTypeError):
        stream_param("join_strays=maybe")


def test_every_stream_parameter_is_part_of_the_cache_key(fetched, capsys):
    assert set(CHANGED_PARAMS) == {f.name for f in dataclasses.fields(StreamParams)}
    assert streams_run(fetched, capsys)[0] == "detected"
    for name, value in CHANGED_PARAMS.items():
        source, summary = streams_run(fetched, capsys, "--param", f"{name}={value}")
        assert source == "detected", name
        assert streams_run(fetched, capsys, "--param", f"{name}={value}")[0] == "cached", name
        if name in ("min_payments", "active_periods"):
            assert summary["gym"] == {"streams": 1, "active": 0, "clients": 1}, name


def test_tables_for_different_parameter_sets_coexist_per_split(fetched, capsys):
    cache = fetched.root / "artifacts" / "streams"
    assert streams_run(fetched, capsys, split="valid")[0] == "detected"
    assert streams_run(fetched, capsys)[0] == "detected"
    assert streams_run(fetched, capsys, "--param", "min_payments=5")[0] == "detected"
    assert streams_run(fetched, capsys, "--param", "min_payments=5", "--param", "active_periods=3")[0] == "detected"
    assert sorted(f.name.split("-")[0] for f in cache.iterdir()) == ["train", "train", "train", "valid"]

    # a sweep back and forth serves each table from the cache, each with its own content
    for _ in range(2):
        source, summary = streams_run(fetched, capsys)
        assert (source, summary["gym"]["active"]) == ("cached", 1)
        source, summary = streams_run(fetched, capsys, "--param", "min_payments=5")
        assert (source, summary["gym"]["active"]) == ("cached", 0)
    assert streams_run(fetched, capsys, split="valid")[0] == "cached"


def test_a_changed_family_table_rebuilds_the_stream_table(fetched, capsys, monkeypatch):
    assert streams_run(fetched, capsys)[1]["gym"]["streams"] == 1
    monkeypatch.delitem(stream_detection.HOME_MCC, "7997")  # gym payments lose their home MCC
    source, summary = streams_run(fetched, capsys)
    assert (source, summary["gym"]["streams"]) == ("detected", 0)


def test_a_changed_detector_source_rebuilds_the_stream_table(fetched, tmp_path):
    # Run the CLI in fresh interpreters from a copy of the package, so the detector source can be edited.
    package = Path(stream_detection.__file__).parent
    copy = tmp_path / "pkg" / package.name
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))

    def run():
        env = {**os.environ, "PYTHONPATH": str(copy.parent)}
        cmd = [sys.executable, "-m", f"{package.name}.cli", "streams", "--split", "train", "--root", str(fetched.root)]
        done = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        return done.stdout.splitlines()[0]

    assert "[detected]" in run()
    assert "[cached]" in run()  # a fresh interpreter derives the same key from unchanged code
    with open(copy / "streams.py", "a") as f:
        f.write("\n# an edit to the detector\n")
    assert "[detected]" in run()
    assert "[cached]" in run()
