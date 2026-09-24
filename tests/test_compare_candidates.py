"""Seam 1: `rf compare` picks a milestone candidate from logged runs by paired bootstrap."""

import csv

import pandas as pd
import pytest

from recurring_family import data

LABELS = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none"]


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_labels(project, split, labels_by_client):
    pd.DataFrame(
        {
            "client_id": list(labels_by_client),
            "cutoff_date": "2026-01-01",
            "target_next_recurring_merchant": list(labels_by_client.values()),
        }
    ).to_csv(project.raw / f"{split}_labels.csv", index=False)


def big_valid(project, n_none, n_gym):
    labels = {f"VN{i:04d}": "none" for i in range(n_none)}
    labels.update({f"VG{i:04d}": "gym" for i in range(n_gym)})
    write_labels(project, "valid", labels)
    return labels


def set_train_majority(project, label):
    ids = pd.read_csv(project.raw / "train_labels.csv", dtype=str)["client_id"]
    other = "cloud" if label != "cloud" else "gym"
    write_labels(project, "train", dict(zip(ids, [label] * 4 + [other] * 2)))


def logged_run(project, label, change):
    """A logged selection-set run of the prior model predicting `label` for every Client."""
    set_train_majority(project, label)
    assert project.run("train", "--model", "prior") == 0
    assert project.run("evaluate", "--model", "prior", "--change", change) == 0
    return read_rows(project.log)[-1]


def compare(project, capsys, *run_ids, out=None):
    capsys.readouterr()
    args = ["compare"] + [a for r in run_ids for a in ("--run", r)]
    if out is not None:
        args += ["--out", str(out)]
    code = project.run(*args)
    return code, capsys.readouterr()


def test_ties_go_to_the_simpler_candidate_listed_first(fetched, capsys):
    big_valid(fetched, n_none=900, n_gym=100)
    # selection: 630 none, 70 gym. all-cloud scores 0, all-gym (2*0.1/1.1)/8 ~ 0.0227: under the margin
    cloud = logged_run(fetched, "cloud", "all cloud")
    gym = logged_run(fetched, "gym", "all gym")
    out = fetched.root / "submissions" / "milestone.md"

    code, printed = compare(fetched, capsys, cloud["run_id"], gym["run_id"], out=out)
    assert code == 0, printed.err
    assert f"winner: {cloud['run_id']}" in printed.out
    assert "tie" in printed.out

    text = out.read_text()
    assert cloud["run_id"] in text and gym["run_id"] in text
    assert "0.0000" in text and "0.0227" in text  # each candidate's macro-F1
    assert "+0.0227" in text  # the paired delta of the later candidate over the simpler one
    assert f"Winner: {cloud['run_id']}" in text


def test_a_clear_improvement_beats_the_simpler_candidates(fetched, capsys):
    big_valid(fetched, n_none=400, n_gym=100)
    # selection: 280 none, 70 gym. all-none beats all-gym by (8/9 - 1/3)/8 ~ 0.069 on every resample
    gym = logged_run(fetched, "gym", "all gym")
    gym_again = logged_run(fetched, "gym", "all gym again")
    none = logged_run(fetched, "none", "all none")

    code, printed = compare(fetched, capsys, gym["run_id"], gym_again["run_id"], none["run_id"])
    assert code == 0, printed.err
    assert f"winner: {none['run_id']}" in printed.out
    assert "improvement" in printed.out


def test_each_candidate_gets_its_macro_f1_and_a_bootstrap_interval(fetched, capsys):
    big_valid(fetched, n_none=400, n_gym=100)
    gym = logged_run(fetched, "gym", "all gym")
    none = logged_run(fetched, "none", "all none")
    out = fetched.root / "cmp.md"
    code, printed = compare(fetched, capsys, gym["run_id"], none["run_id"], out=out)
    assert code == 0, printed.err
    rows = [line for line in out.read_text().splitlines() if line.startswith("|") and "all none" in line]
    assert rows, out.read_text()
    # all-none on 280 none / 70 gym: F1(none) = 2*0.8/1.8, over 8 labels
    assert f"{(1.6 / 1.8) / 8:.4f}" in rows[0]
    assert ".." in rows[0]  # a 95% interval


def test_compare_reads_selection_labels_only(fetched, capsys, monkeypatch):
    big_valid(fetched, n_none=400, n_gym=100)
    gym = logged_run(fetched, "gym", "all gym")
    none = logged_run(fetched, "none", "all none")
    read = []
    real = data.load_labels
    monkeypatch.setattr(data, "load_labels", lambda raw, split: read.append(split) or real(raw, split))
    code, printed = compare(fetched, capsys, gym["run_id"], none["run_id"])
    assert code == 0, printed.err
    assert read == ["selection"]


def test_compare_refuses_sealed_holdout_runs(fetched, capsys):
    big_valid(fetched, n_none=400, n_gym=100)
    gym = logged_run(fetched, "gym", "all gym")
    assert fetched.run("evaluate", "--model", "prior", "--checkpoint", "--change", "checkpoint") == 0
    checkpoint = read_rows(fetched.log)[-1]
    assert checkpoint["split"] == "holdout"
    code, printed = compare(fetched, capsys, gym["run_id"], checkpoint["run_id"])
    assert code != 0
    assert "sealed holdout" in printed.err


def test_compare_refuses_runs_scored_on_different_splits(fetched, capsys):
    big_valid(fetched, n_none=400, n_gym=100)
    gym = logged_run(fetched, "gym", "all gym")
    assert fetched.run("evaluate", "--model", "prior", "--split", "train", "--change", "on train") == 0
    on_train = read_rows(fetched.log)[-1]
    code, printed = compare(fetched, capsys, gym["run_id"], on_train["run_id"])
    assert code != 0
    assert "same split" in printed.err


@pytest.mark.parametrize("problem", ["unknown run", "missing predictions"])
def test_compare_fails_loudly_without_a_runs_log_row_or_predictions(fetched, capsys, problem):
    big_valid(fetched, n_none=400, n_gym=100)
    gym = logged_run(fetched, "gym", "all gym")
    none = logged_run(fetched, "none", "all none")
    other = none["run_id"]
    if problem == "unknown run":
        other = "20990101T000000-000000"
    else:
        (fetched.log.parent / "runs" / f"{other}.csv").unlink()
    out = fetched.root / "cmp.md"
    code, printed = compare(fetched, capsys, gym["run_id"], other, out=out)
    assert code != 0
    assert other in printed.err
    assert not out.exists()


def test_compare_needs_two_distinct_candidates(fetched, capsys):
    big_valid(fetched, n_none=400, n_gym=100)
    gym = logged_run(fetched, "gym", "all gym")
    with pytest.raises(SystemExit):
        compare(fetched, capsys, gym["run_id"])
    with pytest.raises(SystemExit):
        compare(fetched, capsys, gym["run_id"], gym["run_id"])
