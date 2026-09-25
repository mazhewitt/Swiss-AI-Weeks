"""Seam 1: honest evaluation (sealed holdout, leakage guards, bootstrap, CV) through the CLI."""

import csv

import pandas as pd
import pytest

from recurring_family import data, models

LABELS = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none"]


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_labels(project, split, labels_by_client):
    rows = pd.DataFrame(
        {
            "client_id": list(labels_by_client),
            "cutoff_date": "2026-01-01",
            "target_next_recurring_merchant": list(labels_by_client.values()),
        }
    )
    rows.to_csv(project.raw / f"{split}_labels.csv", index=False)


def big_valid(project, n_none=30, n_gym=10, n_cloud=10):
    """A valid label file large enough for a meaningful stratified split and bootstrap."""
    labels = {}
    for i in range(n_none):
        labels[f"VN{i:04d}"] = "none"
    for i in range(n_gym):
        labels[f"VG{i:04d}"] = "gym"
    for i in range(n_cloud):
        labels[f"VC{i:04d}"] = "cloud"
    write_labels(project, "valid", labels)
    return labels


def read_split(project, capsys):
    capsys.readouterr()
    assert project.run("split") == 0
    return pd.read_csv(project.root / "artifacts" / "valid_split.csv", dtype=str)


def set_train_majority(project, label):
    """Six train Clients, four of them `label`: the prior then predicts `label` everywhere."""
    ids = pd.read_csv(project.raw / "train_labels.csv", dtype=str)["client_id"]
    other = "cloud" if label != "cloud" else "gym"
    write_labels(project, "train", dict(zip(ids, [label] * 4 + [other] * 2)))


# --- the selection / sealed-holdout split -------------------------------------


def test_split_is_deterministic_disjoint_complete_and_stratified(fetched, capsys):
    labels = big_valid(fetched)
    first = read_split(fetched, capsys)
    second = read_split(fetched, capsys)
    pd.testing.assert_frame_equal(first, second)

    assert list(first.columns) == ["client_id", "set"]
    assert set(first["set"]) == {"selection", "holdout"}
    assert not first["client_id"].duplicated().any()  # disjoint: each Client in one set
    assert sorted(first["client_id"]) == sorted(labels)  # covers all of valid

    holdout = first.loc[first["set"] == "holdout", "client_id"]
    assert len(holdout) == 15  # 30% of 50
    per_label = pd.Series([labels[c] for c in holdout]).value_counts().to_dict()
    assert per_label == {"none": 9, "gym": 3, "cloud": 3}  # 30% of each label


def test_split_is_about_300_of_1000(fetched, capsys):
    labels = {f"V{i:04d}": LABELS[i % 8] for i in range(1000)}
    write_labels(fetched, "valid", labels)
    split = read_split(fetched, capsys)
    assert (split["set"] == "holdout").sum() == 300
    assert (split["set"] == "selection").sum() == 700


def test_split_does_not_depend_on_file_row_order(fetched, capsys):
    labels = big_valid(fetched)
    first = read_split(fetched, capsys).set_index("client_id")["set"].sort_index()
    write_labels(fetched, "valid", dict(reversed(list(labels.items()))))
    second = read_split(fetched, capsys).set_index("client_id")["set"].sort_index()
    pd.testing.assert_series_equal(first, second)


# --- guards -------------------------------------------------------------------


class LeakyModel(models.PriorModel):
    """Fits on train labels but peeks at a valid label file while doing so."""

    name = "leaky"
    peek = "selection"

    def fit(self, transactions, labels):
        data.load_labels(self.raw, self.peek)
        return super().fit(transactions, labels)


@pytest.fixture
def leaky(fetched, monkeypatch):
    monkeypatch.setattr(LeakyModel, "raw", fetched.raw, raising=False)
    monkeypatch.setattr(LeakyModel, "peek", "selection")
    monkeypatch.setitem(models.MODELS, "leaky", LeakyModel)
    return LeakyModel


@pytest.mark.parametrize("peek", ["selection", "holdout", "valid"])
def test_training_run_that_reads_valid_labels_fails_loudly(fetched, leaky, capsys, peek):
    leaky.peek = peek
    assert fetched.run("train", "--model", "leaky") != 0
    err = capsys.readouterr().err
    assert "valid labels" in err and "training" in err
    assert not (fetched.root / "artifacts" / "leaky.json").exists()


@pytest.mark.parametrize("command", ["train", "cv"])
@pytest.mark.parametrize("peek", ["holdout", "valid"])
def test_refit_with_selection_still_refuses_holdout_and_full_valid_labels(fetched, leaky, capsys, command, peek):
    # The refit loosens the policy to allow selection labels; holdout must stay sealed.
    big_train(fetched)
    big_valid(fetched)
    leaky.peek = peek
    extra = ["--out", str(fetched.root / "oof.csv")] if command == "cv" else []
    assert fetched.run(command, "--model", "leaky", "--with-selection", *extra) != 0
    err = capsys.readouterr().err
    assert "valid labels" in err and "training" in err
    assert not (fetched.root / "artifacts" / "leaky.json").exists()
    assert not (fetched.root / "oof.csv").exists()


def test_training_run_without_valid_reads_succeeds(fetched, leaky):
    leaky.peek = "train"
    assert fetched.run("train", "--model", "leaky") == 0


class HoldoutPeeker(models.PriorModel):
    """Reads sealed-holdout labels while predicting."""

    name = "peeker"

    def predict_proba(self, transactions, clients):
        data.load_labels(self.raw, self.source)
        return super().predict_proba(transactions, clients)


@pytest.mark.parametrize("source", ["holdout", "valid"])
def test_reading_holdout_labels_outside_checkpoint_fails_loudly(fetched, monkeypatch, capsys, source):
    monkeypatch.setattr(HoldoutPeeker, "raw", fetched.raw, raising=False)
    monkeypatch.setattr(HoldoutPeeker, "source", source, raising=False)
    monkeypatch.setitem(models.MODELS, "peeker", HoldoutPeeker)
    assert fetched.run("train", "--model", "peeker") == 0
    assert fetched.run("evaluate", "--model", "peeker") != 0
    assert "sealed holdout" in capsys.readouterr().err
    assert not fetched.log.exists()


class LabelCheater(models.PriorModel):
    """Answers with the true label of every Client whose valid labels it can read while predicting."""

    name = "cheater"

    def predict_proba(self, transactions, clients):
        truth = data.load_labels(self.raw, self.source).set_index("client_id")["target_next_recurring_merchant"]
        rows = [[1.0 if truth.get(c, "none") == l else 0.0 for l in LABELS] for c in clients]
        return pd.DataFrame(rows, index=pd.Index(clients, name="client_id"), columns=LABELS)


@pytest.mark.parametrize(
    ("source", "mode"),
    [
        ("holdout", ["--checkpoint"]),
        ("valid", ["--checkpoint"]),
        ("selection", ["--checkpoint"]),
        ("selection", []),
        ("selection", ["--split", "train"]),
    ],
)
def test_a_model_never_reads_valid_labels_while_predicting(fetched, monkeypatch, capsys, source, mode):
    # Only the scoring step, after predictions are made, may read the labels it scores against;
    # otherwise a model could fake a perfect selection or sealed-holdout score.
    big_valid(fetched)
    monkeypatch.setattr(LabelCheater, "raw", fetched.raw, raising=False)
    monkeypatch.setattr(LabelCheater, "source", source, raising=False)
    monkeypatch.setitem(models.MODELS, "cheater", LabelCheater)
    assert fetched.run("train", "--model", "cheater") == 0
    capsys.readouterr()
    assert fetched.run("evaluate", "--model", "cheater", *mode) != 0
    err = capsys.readouterr().err
    assert f"reading {source} labels" in err and "predict" in err
    assert not fetched.log.exists()
    assert not (fetched.root / "experiments" / "runs").exists()


@pytest.mark.parametrize("source", ["selection", "holdout"])
def test_a_submission_model_never_reads_valid_labels_while_predicting(fetched, monkeypatch, capsys, source):
    # Outside evaluate's scoring step no valid labels are readable, not only inside evaluate.
    big_valid(fetched)
    monkeypatch.setattr(LabelCheater, "raw", fetched.raw, raising=False)
    monkeypatch.setattr(LabelCheater, "source", source, raising=False)
    monkeypatch.setitem(models.MODELS, "cheater", LabelCheater)
    assert fetched.run("train", "--model", "cheater") == 0
    capsys.readouterr()
    assert fetched.run("submit", "--model", "cheater", "--name", "cheat") != 0
    assert f"reading {source} labels" in capsys.readouterr().err
    assert not (fetched.root / "submissions" / "cheat.csv").exists()


@pytest.mark.parametrize("source", ["selection", "holdout", "valid"])
def test_submit_predicts_under_the_predicting_guard(fetched, monkeypatch, capsys, source):
    # submit's prediction runs inside `data.predicting()`: the refusal is the predicting stage's LabelLeak
    big_valid(fetched)
    monkeypatch.setattr(LabelCheater, "raw", fetched.raw, raising=False)
    monkeypatch.setattr(LabelCheater, "source", source, raising=False)
    monkeypatch.setitem(models.MODELS, "cheater", LabelCheater)
    raised = []
    guard = data._guard_label_read

    def recording(split):
        try:
            guard(split)
        except data.LabelLeak as leak:
            raised.append(str(leak))
            raise

    monkeypatch.setattr(data, "_guard_label_read", recording)
    assert fetched.run("train", "--model", "cheater") == 0
    capsys.readouterr()
    assert fetched.run("submit", "--model", "cheater", "--name", "cheat") != 0
    assert len(raised) == 1 and "while a model predicts" in raised[0]
    assert "while a model predicts" in capsys.readouterr().err
    assert not (fetched.root / "submissions" / "cheat.csv").exists()


@pytest.mark.parametrize("split", ["holdout", "valid"])
def test_evaluate_refuses_holdout_or_all_of_valid_without_checkpoint(fetched, split):
    assert fetched.run("train", "--model", "prior") == 0
    with pytest.raises(SystemExit):  # not even an allowed --split choice
        fetched.run("evaluate", "--model", "prior", "--split", split)
    assert not fetched.log.exists()


def test_checkpoint_mode_scores_the_holdout_and_logs_its_own_row_type(fetched, capsys):
    labels = big_valid(fetched)
    holdout = read_split(fetched, capsys).query("set == 'holdout'")["client_id"]
    set_train_majority(fetched, "none")
    assert fetched.run("train", "--model", "prior") == 0
    assert fetched.run("evaluate", "--model", "prior") == 0
    out = fetched.root / "checkpoint_proba.csv"
    assert fetched.run(
        "evaluate", "--model", "prior", "--checkpoint", "--change", "m1 checkpoint", "--proba", str(out)
    ) == 0
    # exactly the sealed-holdout Clients were scored, not the selection set or all of valid
    assert pd.read_csv(out, dtype=str)["client_id"].tolist() == holdout.tolist()

    rows = read_rows(fetched.log)
    assert [r["row_type"] for r in rows] == ["evaluate", "checkpoint"]
    assert [r["split"] for r in rows] == ["selection", "holdout"]
    # holdout: 9 none of 15, all-none prediction -> F1(none) = 2*0.6/1.6
    truth_none = sum(labels[c] == "none" for c in holdout) / len(holdout)
    assert truth_none == pytest.approx(0.6)
    assert float(rows[1]["f1_none"]) == pytest.approx(0.75, abs=1e-4)


def test_evaluate_scores_the_selection_set_by_default(fetched, capsys):
    labels = big_valid(fetched)
    selection = read_split(fetched, capsys).query("set == 'selection'")["client_id"].tolist()
    assert fetched.run("train", "--model", "prior") == 0
    out = fetched.root / "proba.csv"
    assert fetched.run("evaluate", "--model", "prior", "--proba", str(out)) == 0
    assert pd.read_csv(out, dtype=str)["client_id"].tolist() == selection
    row = read_rows(fetched.log)[0]
    assert row["split"] == "selection"
    # train majority none; selection: 21 none of 35 -> F1(none) = 2*0.6/1.6
    assert sum(labels[c] == "none" for c in selection) == 21
    assert float(row["f1_none"]) == pytest.approx(0.75, abs=1e-4)


# --- paired bootstrap against the previous best -------------------------------


def evaluate_with_majority(project, label, change):
    set_train_majority(project, label)
    assert project.run("train", "--model", "prior") == 0
    assert project.run("evaluate", "--model", "prior", "--change", change) == 0
    return read_rows(project.log)[-1]


def test_bootstrap_verdicts_compare_against_the_previous_best(fetched):
    big_valid(fetched, n_none=400, n_gym=100, n_cloud=0)
    # selection: 280 none, 70 gym. all-gym macro = (2*0.2/1.2)/8, all-none = (2*0.8/1.8)/8
    gym = evaluate_with_majority(fetched, "gym", "all gym")
    assert gym["verdict"] == "first" and gym["delta"] == ""

    none = evaluate_with_majority(fetched, "none", "all none")
    assert none["verdict"] == "improvement"
    assert float(none["delta"]) == pytest.approx((8 / 9 - 1 / 3) / 8, abs=1e-4)
    assert none["compared_to"] == gym["run_id"]

    same = evaluate_with_majority(fetched, "none", "all none again")
    assert same["verdict"] == "tie" and float(same["delta"]) == pytest.approx(0.0)
    assert same["compared_to"] == none["run_id"]

    worse = evaluate_with_majority(fetched, "gym", "all gym again")
    assert worse["verdict"] == "worse"  # compared to the best run, not the last one
    assert float(worse["delta"]) == pytest.approx(-(8 / 9 - 1 / 3) / 8, abs=1e-4)
    assert worse["compared_to"] == none["run_id"]


def test_deltas_under_the_tie_margin_are_ties_even_when_consistent(fetched):
    big_valid(fetched, n_none=900, n_gym=100, n_cloud=0)
    # selection: 630 none, 70 gym. all-cloud scores 0; all-gym scores
    # (2*0.1/1.1)/8 ~ 0.0227 < 0.03, and beats all-cloud on every resample
    # (gym share 0.1 +- 0.011), so only the tie margin makes this a tie.
    cloud = evaluate_with_majority(fetched, "cloud", "all cloud")
    gym = evaluate_with_majority(fetched, "gym", "all gym")
    assert float(cloud["macro_f1"]) == 0.0
    assert float(gym["delta"]) == pytest.approx((0.2 / 1.1) / 8, abs=1e-4)
    assert gym["verdict"] == "tie"


def test_large_delta_whose_paired_interval_crosses_zero_is_a_tie(fetched, capsys):
    labels = big_valid(fetched, n_none=10, n_gym=4, n_cloud=0)
    selection = read_split(fetched, capsys).query("set == 'selection'")["client_id"]
    assert sorted(labels[c] for c in selection) == ["gym"] * 3 + ["none"] * 7
    # all-none vs all-gym on 10 Clients: delta = (2*0.7/1.7 - 2*0.3/1.3)/8 ~ 0.045 >= the
    # tie margin, but resampling 10 Clients often flips the none share below 0.5.
    gym = evaluate_with_majority(fetched, "gym", "all gym")
    none = evaluate_with_majority(fetched, "none", "all none")
    assert float(none["delta"]) == pytest.approx((1.4 / 1.7 - 0.6 / 1.3) / 8, abs=1e-4)
    assert float(none["delta"]) >= 0.03
    assert none["compared_to"] == gym["run_id"]
    assert none["verdict"] == "tie"


def scripted(name, answers):
    """A model that predicts a fixed label per Client (and `none` for anyone else)."""

    class Scripted(models.PriorModel):
        def fit(self, transactions, labels):
            return self

        def predict_proba(self, transactions, clients):
            rows = [[1.0 if answers.get(c, "none") == l else 0.0 for l in LABELS] for c in clients]
            return pd.DataFrame(rows, index=pd.Index(clients, name="client_id"), columns=LABELS)

    Scripted.name = name
    return Scripted


def test_bootstrap_is_paired_so_a_small_consistent_gain_is_an_improvement(fetched, capsys, monkeypatch):
    labels = big_valid(fetched, n_none=40, n_gym=20, n_cloud=20)
    selection = read_split(fetched, capsys).query("set == 'selection'")["client_id"]
    by_label = {l: sorted(c for c in selection if labels[c] == l) for l in ["none", "gym", "cloud"]}
    assert {l: len(ids) for l, ids in by_label.items()} == {"none": 28, "gym": 14, "cloud": 14}
    # Both models share many errors, which makes each macro-F1 noisy on 56 Clients:
    # half the gym and cloud Clients are called none, and 8 none Clients gym/cloud.
    base = {c: labels[c] for c in selection}
    for c in by_label["gym"][:7] + by_label["cloud"][:7]:
        base[c] = "none"
    for i, c in enumerate(by_label["none"][:8]):
        base[c] = "cloud" if i % 2 == 0 else "gym"
    # The old model differs only on 4 gym Clients the new one gets right.
    old = {**base, **{c: "none" for c in by_label["gym"][7:11]}}
    monkeypatch.setitem(models.MODELS, "old", scripted("old", old))
    monkeypatch.setitem(models.MODELS, "new", scripted("new", base))

    for name in ["old", "new"]:
        assert fetched.run("train", "--model", name) == 0
        assert fetched.run("evaluate", "--model", name, "--change", name) == 0
    first, second = read_rows(fetched.log)
    # gym F1 6/21 -> 14/25 and none F1 40/66 -> 40/62, over 8 labels: ~0.0392 >= the tie margin.
    # On the same resampled Clients the new model never loses, so the paired interval
    # excludes 0; resampling each model independently would bury the gain in noise.
    assert float(second["delta"]) == pytest.approx((14 / 25 - 6 / 21 + 40 / 62 - 40 / 66) / 8, abs=1e-4)
    assert float(second["delta"]) >= 0.03
    assert second["compared_to"] == first["run_id"]
    assert second["verdict"] == "improvement"


def test_checkpoint_rows_are_not_compared_with_selection_rows(fetched):
    big_valid(fetched)
    evaluate_with_majority(fetched, "none", "selection run")
    assert fetched.run("evaluate", "--model", "prior", "--checkpoint") == 0
    assert read_rows(fetched.log)[-1]["verdict"] == "first"


def run_predictions(project, run_id):
    """Each run's per-Client predictions, committed next to the experiment log."""
    return project.log.parent / "runs" / f"{run_id}.csv"


def test_each_runs_predictions_are_kept_next_to_the_experiment_log(fetched, capsys):
    big_valid(fetched)
    selection = read_split(fetched, capsys).query("set == 'selection'")["client_id"].tolist()
    row = evaluate_with_majority(fetched, "none", "all none")
    saved = pd.read_csv(run_predictions(fetched, row["run_id"]), dtype=str)
    assert saved["client_id"].tolist() == selection
    assert set(saved["predicted"]) == {"none"}
    assert not (fetched.root / "artifacts" / "runs").exists()  # gitignored: would not reach other clones


def test_missing_predictions_of_the_previous_best_fail_loudly(fetched, capsys):
    big_valid(fetched, n_none=400, n_gym=100, n_cloud=0)
    best = evaluate_with_majority(fetched, "none", "all none")
    weaker = evaluate_with_majority(fetched, "gym", "all gym")
    assert weaker["compared_to"] == best["run_id"] and weaker["verdict"] == "worse"
    run_predictions(fetched, best["run_id"]).unlink()
    rows_before = fetched.log.read_text()
    runs_before = sorted(run_predictions(fetched, "x").parent.iterdir())
    capsys.readouterr()

    assert fetched.run("evaluate", "--model", "prior", "--change", "all gym again") != 0
    err = capsys.readouterr().err
    assert best["run_id"] in err and "predictions" in err
    # nothing silently compared against the weaker run, nothing logged
    assert fetched.log.read_text() == rows_before
    assert sorted(run_predictions(fetched, "x").parent.iterdir()) == runs_before


def test_previous_best_is_the_best_row_on_the_same_evaluation_split(fetched, capsys):
    # The ticket-01 row: logged against all of valid, before run ids existed; scored high.
    old_header = "date,change,model,split,macro_f1," + ",".join(f"f1_{l}" for l in LABELS) + ",conclusion\n"
    old_row = "2026-09-24T12:42:56+00:00,old run,prior,valid,0.9000," + ",".join(["0.0000"] * 8) + ",floor\n"
    fetched.log.parent.mkdir(parents=True, exist_ok=True)
    fetched.log.write_text(old_header + old_row)
    big_valid(fetched, n_none=400, n_gym=100, n_cloud=0)

    weak = evaluate_with_majority(fetched, "gym", "all gym")  # selection: (2*0.2/1.2)/8 ~ 0.042
    assert weak["verdict"] == "first"
    set_train_majority(fetched, "none")
    assert fetched.run("train", "--model", "prior") == 0
    assert fetched.run("evaluate", "--model", "prior", "--split", "train", "--change", "train split") == 0
    on_train = read_rows(fetched.log)[-1]
    assert fetched.run("evaluate", "--model", "prior", "--checkpoint", "--change", "checkpoint") == 0
    checkpoint = read_rows(fetched.log)[-1]
    assert float(on_train["macro_f1"]) > float(weak["macro_f1"])
    assert float(checkpoint["macro_f1"]) > float(weak["macro_f1"])
    # Rows on other splits are not candidates, so their predictions are not needed.
    run_predictions(fetched, on_train["run_id"]).unlink()
    run_predictions(fetched, checkpoint["run_id"]).unlink()

    assert fetched.run("evaluate", "--model", "prior", "--change", "all none") == 0
    selection = read_rows(fetched.log)[-1]
    assert selection["split"] == "selection"
    assert selection["compared_to"] == weak["run_id"]
    assert selection["verdict"] == "improvement"


def test_previous_best_predictions_for_other_clients_fail_loudly(fetched, capsys):
    set_train_majority(fetched, "none")
    assert fetched.run("train", "--model", "prior") == 0
    assert fetched.run("evaluate", "--model", "prior", "--split", "train") == 0
    best = read_rows(fetched.log)[-1]
    write_labels(fetched, "train", {"C000001": "none", "C000002": "gym"})  # different Clients now
    capsys.readouterr()
    assert fetched.run("evaluate", "--model", "prior", "--split", "train") != 0
    err = capsys.readouterr().err
    assert best["run_id"] in err and "Clients" in err
    assert len(read_rows(fetched.log)) == 1


def test_log_written_before_this_ticket_is_migrated_not_misaligned(fetched):
    old_header = "date,change,model,split,macro_f1," + ",".join(f"f1_{l}" for l in LABELS) + ",conclusion\n"
    old_row = "2026-09-24T12:42:56+00:00,old run,prior,valid,0.0567," + ",".join(["0.0000"] * 8) + ",floor\n"
    fetched.log.parent.mkdir(parents=True, exist_ok=True)
    fetched.log.write_text(old_header + old_row)
    assert fetched.run("train", "--model", "prior") == 0
    assert fetched.run("evaluate", "--model", "prior", "--change", "new run") == 0
    rows = read_rows(fetched.log)
    assert [(r["change"], r["split"], r["conclusion"]) for r in rows] == [
        ("old run", "valid", "floor"),
        ("new run", "selection", ""),
    ]
    assert rows[1]["verdict"] == "first"  # the old full-valid row is not comparable


# --- refit on train plus the selection set ------------------------------------


def test_refit_on_train_plus_selection_uses_selection_labels_but_not_holdout(fetched, capsys):
    labels = big_valid(fetched, n_none=0, n_gym=50, n_cloud=0)  # every valid Client is gym
    split = read_split(fetched, capsys)
    assert fetched.run("train", "--model", "prior", "--with-selection") == 0
    out = fetched.root / "proba.csv"
    assert fetched.run("evaluate", "--model", "prior", "--split", "train", "--proba", str(out)) == 0
    proba = pd.read_csv(out).iloc[0]
    # train: gym, cloud, none x3, insurance; plus 35 selection gym Clients (holdout's 15 excluded)
    assert (split["set"] == "selection").sum() == 35
    assert proba["gym"] == pytest.approx(36 / 41)
    assert proba["none"] == pytest.approx(3 / 41)
    assert len(labels) == 50


class TransactionRecorder(models.PriorModel):
    """Records whose transactions every fit call was given."""

    name = "recorder"
    fits: list = []

    def fit(self, transactions, labels):
        type(self).fits.append(set(transactions["client_id"]))
        return super().fit(transactions, labels)


@pytest.mark.parametrize("command", ["train", "cv"])
@pytest.mark.parametrize("with_selection", [False, True])
def test_training_fits_on_train_and_selection_transactions_but_never_holdout(
    fetched, monkeypatch, capsys, command, with_selection
):
    monkeypatch.setattr(TransactionRecorder, "fits", [])
    monkeypatch.setitem(models.MODELS, "recorder", TransactionRecorder)
    split = read_split(fetched, capsys)
    selection = set(split.query("set == 'selection'")["client_id"])
    holdout = set(split.query("set == 'holdout'")["client_id"])
    train = set(pd.read_csv(fetched.raw / "train_labels.csv", dtype=str)["client_id"])
    # the fixture has transactions for every valid Client, so either leak would be visible
    valid_tx = set(pd.read_json(fetched.raw / "valid_transactions.jsonl", lines=True, dtype=str)["client_id"])
    assert selection and holdout and valid_tx == selection | holdout

    flags = ["--with-selection"] if with_selection else []
    extra = ["--folds", "2", "--out", str(fetched.root / "oof.csv")] if command == "cv" else []
    assert fetched.run(command, "--model", "recorder", *flags, *extra) == 0

    seen = set().union(*TransactionRecorder.fits)
    assert seen == (train | selection if with_selection else train)
    assert not seen & holdout


def test_model_refit_with_selection_cannot_be_scored_on_the_selection_set(fetched, capsys):
    assert fetched.run("train", "--model", "prior", "--with-selection") == 0
    assert fetched.run("evaluate", "--model", "prior") != 0
    assert "selection" in capsys.readouterr().err
    assert not fetched.log.exists()


# --- 5-fold stratified cross-validation ---------------------------------------


def big_train(project):
    labels = {}
    for i in range(20):
        labels[f"T{i:04d}"] = "none"
    for i in range(10):
        labels[f"T{100 + i:04d}"] = "gym"
    for i in range(7):
        labels[f"T{200 + i:04d}"] = "cloud"
    write_labels(project, "train", labels)
    return labels


def test_cv_saves_out_of_fold_probabilities_fitted_without_each_clients_fold(fetched):
    labels = big_train(fetched)
    out = fetched.root / "oof.csv"
    assert fetched.run("cv", "--model", "prior", "--out", str(out)) == 0
    oof = pd.read_csv(out, dtype={"client_id": str})
    assert list(oof.columns) == ["client_id", "fold", *LABELS]
    assert sorted(oof["client_id"]) == sorted(labels)
    assert sorted(oof["fold"].unique()) == [0, 1, 2, 3, 4]

    truth = oof["client_id"].map(labels)
    for label in ["none", "gym", "cloud"]:  # stratified: each label spread evenly over folds
        counts = oof.loc[truth == label, "fold"].value_counts().reindex(range(5), fill_value=0)
        assert counts.max() - counts.min() <= 1, label

    for _, row in oof.iterrows():
        others = truth[oof["fold"] != row["fold"]]
        freq = others.value_counts(normalize=True)
        for label in LABELS:
            assert row[label] == pytest.approx(freq.get(label, 0.0)), (row["client_id"], label)


class Memoriser(models.PriorModel):
    """Recalls the label of every Client it was fitted on; uniform for anyone else."""

    name = "memoriser"

    def fit(self, transactions, labels):
        self.seen = dict(labels)
        return self

    def predict_proba(self, transactions, clients):
        rows = []
        for c in clients:
            rows.append([1.0 if self.seen.get(c) == l else 0.0 for l in LABELS] if c in self.seen else [1 / 8] * 8)
        return pd.DataFrame(rows, index=pd.Index(clients, name="client_id"), columns=LABELS)


def test_cv_works_for_any_model_and_never_scores_a_client_it_was_fitted_on(fetched, monkeypatch):
    # Leakage trap: a memoriser is perfect in-sample and uniform out-of-fold.
    monkeypatch.setitem(models.MODELS, "memoriser", Memoriser)
    big_train(fetched)
    out = fetched.root / "oof.csv"
    assert fetched.run("cv", "--model", "memoriser", "--out", str(out)) == 0
    oof = pd.read_csv(out)
    assert (oof[LABELS] == 1 / 8).all().all()


def test_cv_with_selection_covers_train_and_selection_but_not_holdout(fetched, capsys):
    train = big_train(fetched)
    big_valid(fetched)
    split = read_split(fetched, capsys)
    out = fetched.root / "oof.csv"
    assert fetched.run("cv", "--model", "prior", "--with-selection", "--out", str(out)) == 0
    oof = pd.read_csv(out, dtype={"client_id": str})
    selection = split.query("set == 'selection'")["client_id"].tolist()
    assert sorted(oof["client_id"]) == sorted([*train, *selection])


def test_cv_is_a_training_run_and_guards_valid_labels(fetched, leaky, capsys):
    big_train(fetched)
    assert fetched.run("cv", "--model", "leaky", "--out", str(fetched.root / "oof.csv")) != 0
    assert "valid labels" in capsys.readouterr().err
