"""Seam 1: the pipeline through the CLI against the fixture data folder."""

import csv
import os

import pandas as pd
import pytest

from conftest import FIXTURE_DATA

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
    # valid truth: none, none, gym, cloud. The prior (train majority = none)
    # predicts all none: F1(none) = 2 * 0.5 * 1 / 1.5 = 2/3; every family 0.
    assert fetched.run("train", "--model", "prior") == 0
    assert fetched.run("evaluate", "--model", "prior", "--change", "prior baseline") == 0

    rows = read_csv_rows(fetched.log)
    assert len(rows) == 1
    row = rows[0]
    assert float(row["macro_f1"]) == pytest.approx((2 / 3) / 8, abs=1e-4)
    assert float(row["f1_none"]) == pytest.approx(2 / 3, abs=1e-4)
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
    # valid truth has one gym among four: precision 0.25, recall 1 -> F1 0.4
    assert float(row["f1_gym"]) == pytest.approx(0.4, abs=1e-4)
    assert float(row["f1_none"]) == 0.0

    assert fetched.run("submit", "--model", "prior", "--name", "gym") == 0
    sub = pd.read_csv(fetched.root / "submissions" / "gym.csv")
    assert set(sub["predicted_next_recurring_merchant"]) == {"gym"}


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
