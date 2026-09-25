"""The Survival Race (`--model survival`): its training rows and its combining rule on hand-made stream
tables, and the model through the CLI against fixture data (Seam 1)."""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from recurring_family import data, models, survival
from recurring_family.config import MERCHANT_FAMILIES
from recurring_family.ranker import FEATURE_COLUMNS
from recurring_family.streams import detect_streams
from recurring_family.survival import (
    DEFAULT_ORDER,
    ORDERS,
    SurvivalModel,
    race_order,
    race_proba,
    race_table,
    training_rows,
)
from test_lgbm_pipeline import LABELS, SPLIT_FILES, read_rows, shop, write_sample_submission, write_transactions
from test_ranker import read_proba, separable_project
from test_ranker_none_model import DAY, train_split, with_decoys
from test_streams import frame as stream_frame
from test_streams import series as stream_series

FAMILIES = [label for label in LABELS if label != "none"]


def table(*streams, order=DEFAULT_ORDER):
    """A race-ordered table from (client_id, family, days_to_next, n_payments[, days_since_last]) tuples;
    `days_since_last` defaults to 0 (paid on the Cutoff)."""
    rows = pd.DataFrame(
        [(*stream, 0.0) if len(stream) == 4 else stream for stream in streams],
        columns=["client_id", "family", "days_to_next", "n_payments", "days_since_last"],
    )
    rows["stream"] = range(len(rows))  # the hand-made stream's position, to read the order back
    return race_order(rows, order)


def rows_of(t, labels):
    keep, target, unexplained = training_rows(t, pd.Series(labels))
    kept = t.loc[keep].assign(target=target[keep])
    return kept, unexplained


def kept_streams(kept, client):
    mine = kept[kept["client_id"] == client]
    return dict(zip(mine["stream"], mine["target"]))


# --- race order -------------------------------------------------------------------------------


def test_the_default_order_gives_an_unprojected_stream_a_monthly_slot():
    assert DEFAULT_ORDER == "monthly-slot"
    t = table(
        ("A", "music", 20.0, 3),  # 0
        ("A", "gym", np.nan, 1, 20.0),  # 1: one payment 20 days ago -> slot 10.4
        ("A", "cloud", 5.0, 4),  # 2
        ("A", "mobile", np.nan, 1, 100.0),  # 3: 100 days ago -> -69.6, rolled 3 months -> 21.6
        ("A", "software", np.nan, 1, 9.6),  # 4: 9.6 days ago -> 20.8
        ("A", "insurance", np.nan, 1, 30.4),  # 5: on the Cutoff exactly -> 0, not rolled
        ("B", "gym", np.nan, 1, 0.0),  # 6
    )
    a = t[t["client_id"] == "A"]
    assert a["stream"].tolist() == [5, 2, 1, 0, 4, 3]
    assert a["order"].tolist() == [0, 1, 2, 3, 4, 5]
    # the slot orders only: the feature stays missing, and next_rank is the place in the race
    assert a.set_index("stream").loc[[5, 1, 4, 3], "days_to_next"].isna().all()
    assert a["next_rank"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert t.loc[t["client_id"] == "B", "order"].tolist() == [0]


def test_a_monthly_slot_tie_goes_to_the_projected_stream_with_more_payments_then_stream_table_order():
    t = table(
        ("A", "music", np.nan, 1, 20.4),  # 0: slot 10.0
        ("A", "gym", 10.0, 3),  # 1
        ("A", "cloud", np.nan, 1, 20.4),  # 2: slot 10.0, after stream 0 in the table
    )
    assert t["stream"].tolist() == [1, 0, 2]


def test_recent_first_orders_the_unprojected_block_by_its_last_payment():
    t = table(
        ("A", "gym", np.nan, 1, 50.0),  # 0
        ("A", "music", 20.0, 3),  # 1
        ("A", "cloud", np.nan, 1, 5.0),  # 2
        ("A", "mobile", np.nan, 2, 50.0),  # 3: ties with 0, more payments
        ("A", "software", np.nan, 1, 50.0),  # 4: ties with 0 exactly, stays after it
        order="recent-first",
    )
    assert t["stream"].tolist() == [1, 2, 3, 0, 4]
    assert t["next_rank"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_every_order_is_reachable_and_an_unknown_one_is_refused():
    t = table(("A", "gym", 3.0, 5), ("A", "music", np.nan, 1, 1.0))
    for order in ORDERS:
        assert len(race_order(t.drop(columns="order"), order)) == 2
    with pytest.raises(ValueError):
        race_order(t, "alphabetical")
    with pytest.raises(ValueError):
        SurvivalModel(order="alphabetical")


def test_unprojected_last_races_projected_payment_with_no_projection_last_and_ties_to_more_payments():
    t = table(
        ("A", "gym", np.nan, 1),  # 0: one payment, no projection
        ("A", "music", 20.0, 3),  # 1
        ("A", "cloud", 5.0, 4),  # 2
        ("A", "mobile", 20.0, 7),  # 3: ties with 1, more payments
        ("A", "software", 20.0, 3),  # 4: ties with 1 exactly, stays after it
        ("B", "gym", np.nan, 2),  # 5
        order="unprojected-last",
    )
    assert t.loc[t["client_id"] == "A", "stream"].tolist() == [2, 3, 1, 4, 0]
    assert t.loc[t["client_id"] == "A", "order"].tolist() == [0, 1, 2, 3, 4]
    assert t.loc[t["client_id"] == "B", "order"].tolist() == [0]


# --- training rows ----------------------------------------------------------------------------


def test_label_on_the_second_stream_trains_the_first_as_a_loser_and_censors_the_rest():
    t = table(("A", "gym", 3.0, 5), ("A", "music", 9.0, 5), ("A", "cloud", 15.0, 5))
    kept, unexplained = rows_of(t, {"A": "music"})
    assert kept_streams(kept, "A") == {0: 0, 1: 1}
    assert unexplained == 0


def test_a_none_client_trains_every_stream_as_a_loser():
    t = table(("A", "gym", 3.0, 5), ("A", "music", 9.0, 5), ("A", "cloud", np.nan, 1))
    kept, _ = rows_of(t, {"A": "none"})
    assert kept_streams(kept, "A") == {0: 0, 1: 0, 2: 0}


def test_a_client_whose_label_family_has_no_candidate_stream_gives_no_rows_and_is_counted():
    t = table(("A", "gym", 3.0, 5), ("A", "music", 9.0, 5), ("B", "cloud", 4.0, 5))
    kept, unexplained = rows_of(t, {"A": "insurance", "B": "cloud", "C": "gym"})  # C has no stream at all
    assert kept_streams(kept, "A") == {}
    assert kept_streams(kept, "B") == {2: 1}
    assert unexplained == 2


def test_under_unprojected_last_streams_with_no_projection_race_last():
    t = table(("A", "gym", np.nan, 1), ("A", "music", 40.0, 3), order="unprojected-last")
    kept, _ = rows_of(t, {"A": "gym"})
    assert kept_streams(kept, "A") == {1: 0, 0: 1}  # music is ahead and lost; gym won last
    kept, _ = rows_of(t, {"A": "music"})
    assert kept_streams(kept, "A") == {1: 1}  # the unprojected gym stream is censored


def test_under_the_default_order_an_unprojected_stream_can_win_ahead_of_a_projected_one():
    t = table(("A", "gym", np.nan, 1, 25.0), ("A", "music", 9.0, 5), ("A", "cloud", 40.0, 5))  # gym slot 5.4
    kept, _ = rows_of(t, {"A": "gym"})
    assert kept_streams(kept, "A") == {0: 1}
    kept, _ = rows_of(t, {"A": "music"})
    assert kept_streams(kept, "A") == {0: 0, 1: 1}


def test_of_two_streams_of_the_label_family_the_first_in_race_order_takes_the_target():
    t = table(("A", "gym", 30.0, 5), ("A", "music", 10.0, 5), ("A", "gym", 20.0, 5))
    kept, _ = rows_of(t, {"A": "gym"})
    assert kept_streams(kept, "A") == {1: 0, 2: 1}  # the later gym stream is censored


def test_a_projection_tie_goes_to_the_stream_with_more_payments():
    t = table(("A", "gym", 10.0, 3), ("A", "music", 10.0, 8))
    kept, _ = rows_of(t, {"A": "gym"})
    assert kept_streams(kept, "A") == {1: 0, 0: 1}
    kept, _ = rows_of(t, {"A": "music"})
    assert kept_streams(kept, "A") == {1: 1}


# --- combining the survival probabilities ---------------------------------------------------------


def test_known_survival_probabilities_give_the_closed_form_race():
    t = table(("A", "gym", 3.0, 5), ("A", "music", 9.0, 5), ("A", "cloud", 15.0, 5), ("B", "mobile", 2.0, 4))
    s = t["stream"].map({0: 0.5, 1: 0.4, 2: 0.2, 3: 0.3}).to_numpy()
    proba = race_proba(t, s, pd.Index(["A", "B", "C"]))
    assert list(proba.columns) == LABELS
    assert proba.index.tolist() == ["A", "B", "C"]
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-15)
    a = proba.loc["A"]
    assert a["gym"] == pytest.approx(0.5)
    assert a["music"] == pytest.approx(0.5 * 0.4)
    assert a["cloud"] == pytest.approx(0.5 * 0.6 * 0.2)
    assert a["none"] == pytest.approx(0.5 * 0.6 * 0.8)
    assert proba.loc["B", "mobile"] == pytest.approx(0.3)
    assert proba.loc["B", "none"] == pytest.approx(0.7)
    # C has no Candidate Stream: certainly none
    assert proba.loc["C", "none"] == 1.0
    assert proba.loc["C", FAMILIES].sum() == 0.0


def test_a_familys_streams_sum():
    t = table(("A", "gym", 3.0, 5), ("A", "music", 9.0, 5), ("A", "gym", 15.0, 5))
    s = t["stream"].map({0: 0.5, 1: 0.4, 2: 0.2}).to_numpy()
    a = race_proba(t, s, pd.Index(["A"])).loc["A"]
    assert a["gym"] == pytest.approx(0.5 + 0.5 * 0.6 * 0.2)
    assert a["music"] == pytest.approx(0.5 * 0.4)
    assert a["none"] == pytest.approx(0.5 * 0.6 * 0.8)
    assert a.sum() == pytest.approx(1.0, abs=1e-15)


def test_rows_sum_to_one_for_any_survival_probabilities():
    rng = np.random.default_rng(0)
    streams = [(f"C{i % 7}", FAMILIES[rng.integers(len(FAMILIES))], float(rng.integers(0, 60)), 3) for i in range(40)]
    t = table(*streams)
    for s in (rng.random(len(t)), np.ones(len(t)), np.zeros(len(t))):
        proba = race_proba(t, s, pd.Index([f"C{i}" for i in range(9)]))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        assert ((proba >= 0) & (proba <= 1)).all().all()


# --- the model --------------------------------------------------------------------------------


def evaluate_proba(project, *extra):
    out = project.root / "proba.csv"
    assert project.run("evaluate", "--model", "survival", "--proba", str(out), *extra) == 0
    return read_proba(out)


def test_train_cv_evaluate_and_submit_with_the_tuned_decision(fetched, capsys):
    train, _ = separable_project(fetched)
    tx = [json.loads(line) for line in (fetched.raw / SPLIT_FILES["test"]).read_text().splitlines()]
    write_transactions(fetched, "test", tx + shop("C000099"))
    write_sample_submission(fetched, ["C000004", "C000008", "C000010", "C000099"])

    out = fetched.root / "oof.csv"
    assert fetched.run("cv", "--model", "survival", "--out", str(out)) == 0
    oof = pd.read_csv(out, dtype={"client_id": str})
    assert sorted(oof["client_id"]) == sorted(train)
    np.testing.assert_allclose(oof[LABELS].sum(axis=1), 1.0, rtol=0, atol=1e-12)

    capsys.readouterr()
    assert fetched.run("train", "--model", "survival", "--decision", "tuned") == 0
    printed = capsys.readouterr().out
    assert "survival race (monthly-slot):" in printed and "left out" in printed
    proba = evaluate_proba(fetched, "--decision", "tuned", "--change", "survival race")
    assert list(proba.columns) == LABELS
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    row = read_rows(fetched.log)[-1]
    # the default race order is named in the log's model column
    assert (row["model"], row["split"]) == ("survival+monthly-slot+tuned", "selection")
    assert float(row["macro_f1"]) >= 0.9, row

    assert fetched.run("submit", "--model", "survival", "--decision", "tuned", "--name", "survival") == 0
    path = fetched.root / "submissions" / "survival.csv"
    sub = pd.read_csv(path, dtype=str).set_index("client_id")["predicted_next_recurring_merchant"]
    assert sub["C000099"] == "none"  # no Candidate Streams
    assert fetched.run("submit", "--check", str(path)) == 0


def test_save_then_load_gives_identical_probabilities(fetched, tmp_path):
    separable_project(fetched)
    transactions, labels = train_split(fetched)
    valid = data.load_transactions(fetched.raw, "valid")
    clients = pd.Index(sorted(valid["client_id"].unique()), name="client_id")

    model = SurvivalModel().fit(transactions, labels)
    assert model.booster is not None and model.n_training_rows > 0
    before = model.predict_proba(valid, clients)
    model.save(tmp_path / "survival.json")
    loaded = SurvivalModel.load(tmp_path / "survival.json")
    pd.testing.assert_frame_equal(loaded.predict_proba(valid, clients), before, check_exact=True)
    assert (loaded.n_training_rows, loaded.n_unexplained) == (model.n_training_rows, model.n_unexplained)
    np.testing.assert_allclose(before.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_training_with_one_outcome_gives_every_stream_a_constant_survival(fetched):
    separable_project(fetched)
    transactions, labels = train_split(fetched)
    valid = data.load_transactions(fetched.raw, "valid")
    clients = pd.Index(sorted(valid["client_id"].unique()), name="client_id")
    everyone_none = pd.Series("none", index=labels.index)
    model = SurvivalModel().fit(transactions, everyone_none)
    assert (model.booster, model.constant) == (None, 0.0)
    proba = model.predict_proba(valid, clients)
    assert (proba["none"] == 1.0).all()


def test_pseudo_labels_and_the_none_model_are_refused(fetched, capsys):
    separable_project(fetched)
    for command in ("train", "cv"):
        for flags in (["--pseudo", "unlabeled"], ["--none-model"], ["--pseudo-weight", "0.5"],
                      ["--pseudo-min-payments", "4"]):
            with pytest.raises(SystemExit):
                fetched.run(command, "--model", "survival", *flags)
            err = capsys.readouterr().err
            assert "--pseudo" in err or "--none-model" in err
    assert not (fetched.root / "artifacts" / "survival.json").exists()


def test_training_and_the_tuned_decision_read_train_labels_only(fetched, monkeypatch):
    separable_project(fetched)
    assert fetched.run("split") == 0
    read = []
    load_labels = data.load_labels

    def recording(raw_dir, split):
        read.append(split)
        return load_labels(raw_dir, split)

    monkeypatch.setattr(data, "load_labels", recording)
    assert fetched.run("train", "--model", "survival", "--decision", "tuned") == 0
    assert fetched.run("cv", "--model", "survival", "--out", str(fetched.root / "oof.csv")) == 0
    assert read and set(read) == {"train"}


# --- ADR 0001: the stream table only ------------------------------------------------------------


def test_decoys_leave_the_features_the_training_rows_and_the_probabilities_unchanged(fetched):
    separable_project(fetched)
    transactions, labels = train_split(fetched)
    valid = data.load_transactions(fetched.raw, "valid")
    clients = pd.Index(sorted(valid["client_id"].unique()), name="client_id")
    model = SurvivalModel().fit(transactions, labels)
    before_table = race_table(detect_streams(transactions))
    before_proba = model.predict_proba(valid, clients)

    decoyed, decoyed_valid = with_decoys(fetched, "train"), with_decoys(fetched, "valid")
    assert len(decoyed) > len(transactions) and len(decoyed_valid) > len(valid)
    after_table = race_table(detect_streams(decoyed))
    pd.testing.assert_frame_equal(after_table, before_table)
    assert list(after_table.columns) == ["client_id", *FEATURE_COLUMNS, "order"]
    pd.testing.assert_frame_equal(model.predict_proba(decoyed_valid, clients), before_proba)
    # a model fitted on the decoyed history is the same model
    refit = SurvivalModel().fit(decoyed, labels)
    pd.testing.assert_frame_equal(refit.predict_proba(decoyed_valid, clients), before_proba)


# --- follow-up tests (fix round 1) -------------------------------------------------------------


def test_none_clients_are_never_counted_unexplained():
    t = table(("A", "gym", 3.0, 5), ("B", "music", 4.0, 5))
    _, unexplained = rows_of(t, {"A": "none", "C": "none", "B": "cloud"})  # C has no stream at all
    assert unexplained == 1  # B only


def hand_race_table(n_clients=12, seed=0):
    """A race-ordered table with every feature column: three streams per Client."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_clients):
        for k in range(3):
            row = {c: float(rng.random()) for c in FEATURE_COLUMNS if c != "family"}
            row["client_id"] = f"C{i:02d}"
            row["family"] = FAMILIES[(i + k) % len(FAMILIES)]
            row["days_to_next"] = float(rng.integers(0, 60))
            row["n_payments"] = float(rng.integers(1, 9))
            rows.append(row)
    out = pd.DataFrame(rows)
    out["family"] = pd.Categorical(out["family"], categories=FAMILIES)
    return race_order(out[["client_id", *FEATURE_COLUMNS]])


@pytest.fixture
def fit_spy(monkeypatch):
    seen = []
    fit = lgb.LGBMClassifier.fit

    def spying(self, x, y, *args, **kwargs):
        seen.append((x.copy(), np.asarray(y).copy()))
        return fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(lgb.LGBMClassifier, "fit", spying)
    return seen


def test_the_fit_sees_exactly_the_kept_rows_and_their_targets(monkeypatch, fit_spy):
    t = hand_race_table()
    clients = sorted(t["client_id"].unique())
    labels = pd.Series(
        {c: "none" if i % 3 == 0 else t.loc[t["client_id"] == c, "family"].astype(str).iloc[i % 3]
         for i, c in enumerate(clients)}
    )
    monkeypatch.setattr(survival, "_streams_of", lambda transactions, clients: None)
    monkeypatch.setattr(survival, "race_table", lambda streams, order: t)
    keep, target, _ = training_rows(t, labels)
    assert 0 < keep.sum() < len(t) and len(set(target[keep])) == 2

    model = SurvivalModel().fit(pd.DataFrame({"client_id": labels.index}), labels)
    [(x, y)] = fit_spy
    pd.testing.assert_frame_equal(x, t.loc[keep, FEATURE_COLUMNS])
    np.testing.assert_array_equal(y, target[keep])
    assert model.n_training_rows == keep.sum()


def test_with_decoys_the_fit_sees_the_undecoyed_rows(fetched, fit_spy):
    separable_project(fetched)
    transactions, labels = train_split(fetched)
    expected = race_table(detect_streams(transactions))
    keep, target, _ = training_rows(expected, labels)

    SurvivalModel().fit(with_decoys(fetched, "train"), labels)
    [(x, y)] = fit_spy
    pd.testing.assert_frame_equal(
        x.reset_index(drop=True), expected.loc[keep, FEATURE_COLUMNS].reset_index(drop=True)
    )
    np.testing.assert_array_equal(y, target[keep])


def test_save_then_load_keeps_a_constant_model(fetched, tmp_path):
    separable_project(fetched)
    transactions, labels = train_split(fetched)
    valid = data.load_transactions(fetched.raw, "valid")
    clients = pd.Index(sorted(valid["client_id"].unique()), name="client_id")
    model = SurvivalModel().fit(transactions, pd.Series("none", index=labels.index))
    model.save(tmp_path / "survival.json")
    loaded = SurvivalModel.load(tmp_path / "survival.json")
    assert (loaded.booster, loaded.constant) == (None, 0.0)
    pd.testing.assert_frame_equal(loaded.predict_proba(valid, clients), model.predict_proba(valid, clients))

    SurvivalModel(None, 0.37).save(tmp_path / "constant.json")
    assert SurvivalModel.load(tmp_path / "constant.json").constant == 0.37


def test_train_tuned_and_cv_outputs_do_not_depend_on_valid_labels(fetched):
    separable_project(fetched)
    assert fetched.run("split") == 0
    artifacts = fetched.root / "artifacts"
    outputs = [artifacts / "survival.json", artifacts / "survival.decision.json", fetched.root / "oof.csv"]

    def run():
        assert fetched.run("train", "--model", "survival", "--decision", "tuned") == 0
        assert fetched.run("cv", "--model", "survival", "--out", str(fetched.root / "oof.csv")) == 0
        return [path.read_bytes() for path in outputs]

    before = run()
    labels = pd.read_csv(fetched.raw / "valid_labels.csv", dtype=str)
    assert (labels["target_next_recurring_merchant"] != "none").any()
    labels["target_next_recurring_merchant"] = "none"
    labels.to_csv(fetched.raw / "valid_labels.csv", index=False)
    assert run() == before


def test_race_table_orders_a_real_stream_table_by_projected_payment():
    streams = detect_streams(stream_frame(stream_series("A", "gym membership", "7997", 66, "2025-06-10", 6)))
    assert len(streams) == 1
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    extra = pd.concat([streams] * 3, ignore_index=True)
    extra["client_id"] = "X"
    extra["next_payment"] = [cutoff + d * DAY for d in (30, 10, 20)]
    t = race_table(pd.concat([streams, extra], ignore_index=True))
    x = t[t["client_id"] == "X"]
    assert x["days_to_next"].tolist() == [10.0, 20.0, 30.0]
    assert x["order"].tolist() == [0, 1, 2]


def test_race_proba_follows_the_requested_index_with_duplicates_and_unknown_clients():
    t = table(("A", "gym", 3.0, 5), ("A", "music", 9.0, 5), ("B", "mobile", 2.0, 4))
    s = t["stream"].map({0: 0.5, 1: 0.4, 2: 0.3}).to_numpy()
    proba = race_proba(t, s, pd.Index(["Z", "B", "A", "B"]))
    assert proba.index.tolist() == ["Z", "B", "A", "B"]
    pd.testing.assert_series_equal(proba.iloc[1], proba.iloc[3])
    assert proba.iloc[0]["none"] == 1.0
    assert proba.iloc[1]["mobile"] == pytest.approx(0.3)
    assert proba.iloc[2]["music"] == pytest.approx(0.2)


def test_save_then_load_with_a_family_absent_from_the_fit(fetched, tmp_path):
    separable_project(fetched)
    transactions, labels = train_split(fetched)
    with_music = set(detect_streams(transactions).query("family == 'music'")["client_id"])
    fit_labels = labels[~labels.index.isin(with_music)]
    assert "music" not in set(fit_labels) and len(set(fit_labels)) > 2
    valid = data.load_transactions(fetched.raw, "valid")
    clients = pd.Index(sorted(valid["client_id"].unique()), name="client_id")
    model = SurvivalModel().fit(transactions, fit_labels)
    before = model.predict_proba(valid, clients)
    assert (before["music"] > 0).any()  # valid has music streams the fit never saw
    model.save(tmp_path / "survival.json")
    pd.testing.assert_frame_equal(SurvivalModel.load(tmp_path / "survival.json").predict_proba(valid, clients), before)


def test_a_certain_survivor_leaves_the_streams_behind_it_nothing():
    t = table(("A", "gym", 3.0, 5), ("A", "music", 9.0, 5))
    proba = race_proba(t, np.array([1.0, 1.0]), pd.Index(["A"]))
    assert proba.loc["A", "gym"] == pytest.approx(1.0, abs=1e-12)
    assert proba.loc["A", "music"] == pytest.approx(0.0, abs=1e-12)
    assert proba.loc["A", "none"] == pytest.approx(0.0, abs=1e-12)


# --- fix round 2 ------------------------------------------------------------------------------------


def test_unexplained_counts_only_clients_whose_transactions_the_fit_was_given(fetched):
    separable_project(fetched)
    transactions, labels = train_split(fetched)
    everyone = SurvivalModel().fit(transactions, labels).n_unexplained
    # labelled with a family, but their transactions are not passed in: not counted
    dropped = set(labels[labels != "none"].index[:5])
    subset = transactions[~transactions["client_id"].isin(dropped)]
    assert SurvivalModel().fit(subset, labels).n_unexplained == everyone
    # a Client with transactions but no Candidate Stream and a family label still counts
    with_streams = set(detect_streams(transactions)["client_id"])
    shopper = next(c for c in labels[labels == "none"].index if c not in with_streams)
    assert shopper in set(subset["client_id"])
    relabelled = labels.copy()
    relabelled[shopper] = "gym"
    assert SurvivalModel().fit(subset, relabelled).n_unexplained == everyone + 1


def test_the_race_order_is_saved_and_a_model_saved_before_it_was_a_setting_races_unprojected_last(fetched, tmp_path):
    separable_project(fetched)
    transactions, labels = train_split(fetched)
    model = SurvivalModel(order="recent-first").fit(transactions, labels)
    model.save(tmp_path / "survival.json")
    assert SurvivalModel.load(tmp_path / "survival.json").order == "recent-first"
    saved = json.loads((tmp_path / "survival.json").read_text())
    del saved["order"]
    (tmp_path / "old.json").write_text(json.dumps(saved))
    assert SurvivalModel.load(tmp_path / "old.json").order == "unprojected-last"
    assert SurvivalModel().order == DEFAULT_ORDER


# --- fix round 3: the race order through the CLI, and the order's own tests -----------------------


class OrderRecorder(SurvivalModel):
    """Records the race order of every model the CLI makes."""

    orders: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).orders.append(self.order)


@pytest.mark.parametrize("order", ORDERS)
def test_race_order_reaches_the_model_and_every_fold_of_the_tuned_decision(fetched, monkeypatch, order):
    separable_project(fetched)
    monkeypatch.setattr(OrderRecorder, "orders", [])
    monkeypatch.setitem(models.MODELS, "survival", OrderRecorder)
    assert fetched.run("train", "--model", "survival", "--race-order", order, "--decision", "tuned", "--folds", "3") == 0
    assert OrderRecorder.orders == [order] * 4  # the model, then one per fold
    artifacts = fetched.root / "artifacts"
    assert json.loads((artifacts / "survival.json").read_text())["order"] == order
    assert json.loads((artifacts / "survival.meta.json").read_text())["race_order"] == order
    assert fetched.run("cv", "--model", "survival", "--race-order", order, "--out", str(fetched.root / "oof.csv")) == 0
    assert OrderRecorder.orders[4:] == [order] * 5


def test_without_race_order_the_cli_races_by_the_default_order(fetched, monkeypatch):
    separable_project(fetched)
    monkeypatch.setattr(OrderRecorder, "orders", [])
    monkeypatch.setitem(models.MODELS, "survival", OrderRecorder)
    assert fetched.run("train", "--model", "survival") == 0
    assert OrderRecorder.orders == [DEFAULT_ORDER]


def test_race_order_applies_to_the_survival_model_only(fetched, capsys):
    separable_project(fetched)
    for command in ("train", "cv"):
        for model in ("ranker", "blend", "rules", "lgbm", "prior"):
            with pytest.raises(SystemExit):
                fetched.run(command, "--model", model, "--race-order", "monthly-slot")
            assert "--race-order" in capsys.readouterr().err
        with pytest.raises(SystemExit):
            fetched.run(command, "--model", "survival", "--race-order", "alphabetical")


@pytest.mark.parametrize(
    ("order", "logged"),
    [("unprojected-last", "survival"), ("recent-first", "survival+recent-first"), ("monthly-slot", "survival+monthly-slot")],
)
def test_the_log_names_the_race_order_except_unprojected_last(fetched, order, logged):
    separable_project(fetched)
    assert SurvivalModel(order=order).variant == logged.removeprefix("survival")
    assert fetched.run("train", "--model", "survival", "--race-order", order) == 0
    assert fetched.run("evaluate", "--model", "survival") == 0
    assert read_rows(fetched.log)[-1]["model"] == logged


def test_a_rolled_monthly_slot_is_the_first_slot_after_the_cutoff():
    # 40 days ago -> -9.6, rolled ONE month -> 20.8: ahead of a projection at 25 days
    rows = pd.DataFrame(
        [("A", "music", 25.0, 3, 0.0), ("A", "gym", np.nan, 1, 40.0)],
        columns=["client_id", "family", "days_to_next", "n_payments", "days_since_last"],
    )
    assert race_order(rows)["family"].tolist() == ["gym", "music"]
    slot = survival._monthly_slot(pd.Series([10.0, 30.4, 40.0, 60.8, 100.0]))
    np.testing.assert_allclose(slot, [20.4, 0.0, 20.8, 0.0, 21.6], atol=1e-9)


def hand_candidates():
    """Three Candidate Streams of Client A as `ranker.candidates` would give them (next_rank is the
    ranker's own, unprojected-last)."""
    rows = pd.DataFrame(
        [("A", "gym", np.nan, 1.0, 25.0, 2.0), ("A", "music", 9.0, 5.0, 3.0, 1.0), ("A", "cloud", np.nan, 1.0, 2.0, 3.0)],
        columns=["client_id", "family", "days_to_next", "n_payments", "days_since_last", "next_rank"],
    )
    rows["family"] = pd.Categorical(rows["family"], categories=list(MERCHANT_FAMILIES))
    for column in FEATURE_COLUMNS:
        if column not in rows:
            rows[column] = 0.0
    return rows


RACES = {  # order -> the race, by family
    "unprojected-last": ["music", "gym", "cloud"],
    "recent-first": ["music", "cloud", "gym"],
    "monthly-slot": ["gym", "music", "cloud"],  # gym's slot 5.4, cloud's 28.4
}


@pytest.mark.parametrize("order", ORDERS)
def test_predict_races_by_the_models_order_and_feeds_next_rank_as_the_place_in_the_race(monkeypatch, order):
    monkeypatch.setattr(survival, "_streams_of", lambda transactions, clients: None)
    monkeypatch.setattr(survival, "candidates", lambda streams, cutoff=None: hand_candidates())
    model = SurvivalModel(constant=0.5, order=order)
    seen = []
    monkeypatch.setattr(model, "survival", lambda t: seen.append(t.copy()) or np.full(len(t), 0.5))
    proba = model.predict_proba(pd.DataFrame(), pd.Index(["A"]))
    [t] = seen
    assert t["family"].astype(str).tolist() == RACES[order]
    assert t["next_rank"].tolist() == [1.0, 2.0, 3.0]
    # the slot orders only: the unprojected streams' days_to_next stays missing at predict time too
    assert t.loc[t["family"].astype(str).isin(["gym", "cloud"]), "days_to_next"].isna().all()
    first = RACES[order][0]
    assert proba.loc["A", first] == pytest.approx(0.5)  # the first in the race gets s, not s(1-s)


@pytest.mark.parametrize(("order", "rows"), [("unprojected-last", 2), ("recent-first", 3), ("monthly-slot", 1)])
def test_fit_races_by_the_models_order(monkeypatch, order, rows):
    monkeypatch.setattr(survival, "_streams_of", lambda transactions, clients: None)
    monkeypatch.setattr(survival, "candidates", lambda streams, cutoff=None: hand_candidates())
    seen = []
    fit = survival.lgb.LGBMClassifier.fit

    def spying(self, x, y, *args, **kwargs):
        seen.append(x.copy())
        return fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(survival.lgb.LGBMClassifier, "fit", spying)
    model = SurvivalModel(order=order).fit(pd.DataFrame({"client_id": ["A"]}), pd.Series({"A": "gym"}))
    assert model.n_training_rows == rows  # gym's place in the race: every stream up to it trains
    if seen:  # the fit sees next_rank as the place in the race too
        assert seen[0]["next_rank"].tolist() == [float(i + 1) for i in range(rows)]
