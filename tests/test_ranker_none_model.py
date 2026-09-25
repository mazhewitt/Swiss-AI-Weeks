"""The Stream Ranker with its Client-level `none` model (`--none-model`): through the CLI against fixture
data (Seam 1), the stream module's member payments (Seam 2), the combining rule on hand-set inputs, and
every feature pinned on a hand-made history."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from recurring_family import data
from recurring_family.none_model import FEATURE_COLUMNS, RANKER_NONE, NoneModel, client_features, combine
from recurring_family.ranker import RankerModel, pseudo_examples
from recurring_family.streams import detect_stream_payments, detect_streams
from test_lgbm_pipeline import (
    LABELS,
    SPLIT_FILES,
    read_rows,
    separable_split,
    series,
    shop,
    write_sample_submission,
    write_transactions,
)
from test_ranker import evaluate_proba, separable_project, valid_rows
from test_ranker_pseudo_labels import SHIFTED as SHIFTED_DAY
from test_ranker_pseudo_labels import pooled_project
from test_streams import frame
from test_streams import series as stream_series
from test_streams import tx as stream_tx

DAY = pd.Timedelta(days=1)
LABEL_COLUMN = "target_next_recurring_merchant"
SHIFTED = pd.Timestamp(SHIFTED_DAY, tz="UTC")


def saved_ranker(project):
    return json.loads((project.root / "artifacts" / "ranker.json").read_text())


def train_split(project):
    transactions = data.load_transactions(project.raw, "train")
    labels = data.load_labels(project.raw, "train").set_index("client_id")[LABEL_COLUMN]
    return transactions, labels


# --- combining the ranker with the `none` model -----------------------------------------------


def test_each_family_keeps_its_share_of_the_ranker_and_none_comes_from_the_none_model():
    ranker = pd.DataFrame(0.0, index=pd.Index(["A", "B", "C", "D"], name="client_id"), columns=LABELS)
    ranker.loc["A", ["gym", "cloud", "none"]] = [0.5, 0.25, 0.25]
    ranker.loc["B", ["music", "none"]] = [0.2, 0.8]
    ranker.loc["C", "none"] = 1.0  # no candidates
    ranker.loc["D", ["mobile", "none"]] = [0.6, 0.4]
    p_none = pd.Series({"A": 0.4, "B": 0.1, "C": 0.3})  # D is not scored by the none model

    out = combine(ranker, p_none)
    assert list(out.columns) == LABELS
    assert np.allclose(out.sum(axis=1), 1.0)
    # A: gym has two thirds of the family scores, cloud one third; together they get 1 - 0.4
    assert out.loc["A", "gym"] == pytest.approx(0.6 * 2 / 3)
    assert out.loc["A", "cloud"] == pytest.approx(0.6 * 1 / 3)
    assert out.loc["A", "none"] == pytest.approx(0.4)
    assert out.loc["B", "music"] == pytest.approx(0.9)
    # C has no family score to share: it stays none; D keeps the ranker's row
    assert out.loc["C", "none"] == 1.0
    pd.testing.assert_series_equal(out.loc["D"], ranker.loc["D"])


# --- the model through the CLI -------------------------------------------------------------------


def test_train_cv_evaluate_and_submit_with_the_none_model_and_the_tuned_decision(fetched, capsys):
    train, _ = pooled_project(fetched)
    tx = [json.loads(line) for line in (fetched.raw / SPLIT_FILES["test"]).read_text().splitlines()]
    write_transactions(fetched, "test", tx + shop("C000099"))
    write_sample_submission(fetched, ["C000004", "C000008", "C000010", "C000099"])
    pseudo = ("--pseudo", "unlabeled")

    out = fetched.root / "oof.csv"
    assert fetched.run("cv", "--model", "ranker", "--none-model", *pseudo, "--out", str(out)) == 0
    oof = pd.read_csv(out, dtype={"client_id": str})
    assert sorted(oof["client_id"]) == sorted(train)  # real-labelled Clients only
    assert np.allclose(oof[LABELS].sum(axis=1), 1.0)

    assert fetched.run("train", "--model", "ranker", "--none-model", "--decision", "tuned", *pseudo) == 0
    proba = evaluate_proba(fetched, "--decision", "tuned", "--change", "none model")
    assert list(proba.columns) == LABELS
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all().all()
    row = read_rows(fetched.log)[-1]
    assert (row["model"], row["split"]) == ("ranker+none+pseudo+tuned", "selection")
    assert float(row["macro_f1"]) >= 0.9, row

    assert fetched.run("submit", "--model", "ranker", "--decision", "tuned", "--name", "none-model") == 0
    path = fetched.root / "submissions" / "none-model.csv"
    sub = pd.read_csv(path, dtype=str).set_index("client_id")["predicted_next_recurring_merchant"]
    assert sub["C000099"] == "none"  # no Candidate Streams
    assert fetched.run("submit", "--check", str(path)) == 0


def test_the_none_model_sets_p_none_of_every_client_with_candidate_streams(fetched):
    separable_split(fetched, "train", "T", 12, np.random.default_rng(3))
    separable_split(fetched, "valid", "V", 8, np.random.default_rng(4))
    assert fetched.run("train", "--model", "ranker") == 0
    plain = evaluate_proba(fetched)
    assert fetched.run("train", "--model", "ranker", "--none-model") == 0
    with_none = evaluate_proba(fetched)

    families = [label for label in LABELS if label != "none"]
    has_family = plain[families].sum(axis=1) > 0
    assert has_family.any() and not has_family.all()
    assert not np.allclose(with_none.loc[has_family, "none"], plain.loc[has_family, "none"])
    # the families keep their shares of the ranker's family probabilities; Clients without candidates stay none
    def shares(proba):
        fam = proba.loc[has_family, families]
        return fam.div(fam.sum(axis=1), axis=0).to_numpy()

    np.testing.assert_allclose(shares(with_none), shares(plain), atol=1e-12)
    pd.testing.assert_frame_equal(with_none.loc[~has_family], plain.loc[~has_family])


def test_save_then_load_gives_identical_probabilities(fetched, tmp_path):
    pooled_project(fetched)
    transactions, labels = train_split(fetched)
    valid = data.load_transactions(fetched.raw, "valid")
    clients = pd.Index(sorted(valid["client_id"].unique()), name="client_id")

    model = RankerModel(none_model=NoneModel()).fit(transactions, labels)
    before = model.predict_proba(valid, clients)
    model.save(tmp_path / "ranker.json")
    loaded = RankerModel.load(tmp_path / "ranker.json")
    assert loaded.variant == "+none"
    pd.testing.assert_frame_equal(loaded.predict_proba(valid, clients), before, check_exact=True)
    assert np.allclose(before.sum(axis=1), 1.0)


def test_without_the_flag_the_saved_ranker_has_no_none_model(fetched):
    pooled_project(fetched)
    assert fetched.run("train", "--model", "ranker") == 0
    assert sorted(saved_ranker(fetched)) == ["booster", "constant", "features", "model"]
    assert fetched.run("evaluate", "--model", "ranker") == 0
    assert read_rows(fetched.log)[-1]["model"] == "ranker"


def test_the_none_model_applies_to_the_ranker_only(fetched):
    pooled_project(fetched)
    for model in ("blend", "rules", "lgbm"):
        with pytest.raises(SystemExit):
            fetched.run("train", "--model", model, "--none-model")


# --- what the none model is fitted on ------------------------------------------------------------


def test_pseudo_labelled_clients_never_reach_the_none_models_fit(fetched):
    train, unlabeled = pooled_project(fetched)
    assert fetched.run("train", "--model", "ranker", "--none-model", "--pseudo", "unlabeled", "--pseudo", "train") == 0
    fitted_on = saved_ranker(fetched)["none_model"]["fitted_on_clients"]

    transactions, _ = train_split(fetched)
    with_streams = set(detect_streams(transactions)["client_id"])
    assert sorted(fitted_on) == sorted(c for c in train if c in with_streams)
    assert not set(fitted_on) & set(unlabeled)
    assert len(fitted_on) < len(train)  # Clients without Candidate Streams are not rows either


def test_no_clients_label_reaches_its_own_none_model_row(fetched):
    train, _ = pooled_project(fetched)
    transactions, labels = train_split(fetched)

    def rows(truth):
        model = RankerModel(none_model=NoneModel())
        return model.none_training_features(transactions, truth)

    before = rows(labels)
    assert list(before.columns) == [RANKER_NONE, *FEATURE_COLUMNS]
    for client in [c for c in labels.index if labels[c] == "gym"][:2]:
        flipped = labels.copy()
        flipped[client] = "none"
        after = rows(flipped)
        # the Client's own row, its cross-fitted ranker score included, ignores its own label ...
        pd.testing.assert_series_equal(after.loc[client], before.loc[client])
        # ... while its label does reach other Clients' ranker scores
        assert not np.allclose(after[RANKER_NONE], before[RANKER_NONE])


def test_the_none_models_training_rows_are_the_client_features_at_the_cutoff_and_a_cross_fitted_ranker_none(fetched):
    _, unlabeled = pooled_project(fetched)
    transactions, labels = train_split(fetched)
    pseudo = pseudo_examples(
        detect_streams(data.load_transactions(fetched.raw, "unlabeled"), SHIFTED), pd.Series(unlabeled), SHIFTED
    )
    model = RankerModel(pseudo=pseudo, pseudo_weight=0.3, none_model=NoneModel())
    x = model.none_training_features(transactions, labels)

    assert list(x.columns) == [RANKER_NONE, *FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(x[FEATURE_COLUMNS], client_features(*detect_stream_payments(transactions)))
    # RANKER_NONE is the ranker's own `none`, each Client's from a ranker fitted on the other four of
    # five shuffled folds of the labelled Clients, with every Pseudo-Labelled row at the model's weight
    expected = pd.Series(np.nan, index=labels.index.astype(str))
    for fit_idx, score_idx in KFold(n_splits=5, shuffle=True, random_state=0).split(labels.index):
        ranker = RankerModel(pseudo=pseudo, pseudo_weight=0.3).fit(transactions, labels.iloc[fit_idx])
        expected.iloc[score_idx] = ranker.predict_proba(transactions, labels.index[score_idx])["none"].to_numpy()
    np.testing.assert_allclose(x[RANKER_NONE].to_numpy(), expected.reindex(x.index).to_numpy(), rtol=0, atol=1e-12)

    model.fit(transactions, labels)
    assert model.none_model.fitted_on_clients == list(x.index)


def test_the_served_p_none_is_the_none_models_prediction_on_the_client_features_and_the_rankers_own_none(fetched):
    separable_project(fetched)
    transactions, labels = train_split(fetched)
    valid = data.load_transactions(fetched.raw, "valid")
    clients = pd.Index(sorted(valid["client_id"].unique()), name="client_id")
    model = RankerModel(none_model=NoneModel()).fit(transactions, labels)
    served = model.predict_proba(valid, clients)

    # the none model's input, by hand: the Client features at the Cutoff plus the plain ranker's `none`
    plain = RankerModel(model.booster, model.constant).predict_proba(valid, clients)
    x = client_features(*detect_stream_payments(valid))
    x.insert(0, RANKER_NONE, plain["none"].reindex(x.index))
    expected = model.none_model.predict(x)
    assert 0 < len(x) < len(clients)
    np.testing.assert_allclose(served.loc[x.index, "none"], expected, rtol=0, atol=1e-12)
    # the fitted none model reads RANKER_NONE: an unknown or flipped ranker `none` would move P(`none`)
    for wrong in (np.nan, 1.0 - x[RANKER_NONE]):
        assert not np.allclose(model.none_model.predict(x.assign(**{RANKER_NONE: wrong})), expected)


# --- ADR 0001: stream member payments and matched refunds only -------------------------------------


def test_decoys_and_unmatched_refunds_leave_the_none_model_features_and_probabilities_unchanged(fetched):
    separable_split(fetched, "train", "T", 12, np.random.default_rng(3))
    separable_split(fetched, "valid", "V", 8, np.random.default_rng(4))
    assert fetched.run("train", "--model", "ranker", "--none-model") == 0
    before_proba = evaluate_proba(fetched)
    before = client_features(*detect_stream_payments(data.load_transactions(fetched.raw, "valid")))

    rows = valid_rows(fetched)
    extra = []
    for client in sorted({r["client_id"] for r in rows}):
        for i, (description, mcc, amount) in enumerate(
            [("digital order", "5732", 6.8), ("merchant charge", "7997", 66.0), ("card purchase", "6300", 109.0)]
        ):
            extra += series(client, description, mcc, amount, f"2025-{8 + i:02d}-14", 3, every_days=31)
        # a refund at a stream amount with a decoy description, and one of a shop payment
        refund = series(client, "digital order", "7997", 66.0, "2025-12-20", 1)[0]
        extra.append({**refund, "type": "refund", "direction": "in"})
        extra.append({**shop(client, "2025-12-21")[0], "type": "refund", "direction": "in"})
    write_transactions(fetched, "valid", rows + extra)

    after = client_features(*detect_stream_payments(data.load_transactions(fetched.raw, "valid")))
    pd.testing.assert_frame_equal(after, before)
    pd.testing.assert_frame_equal(evaluate_proba(fetched), before_proba)


def with_decoys(project, split):
    """The split's transactions plus, for every Client, Decoy Transactions on stream MCCs at stream
    amounts, a refund with a decoy description at a stream amount and a refund of a shop payment.
    The Decoys sit half a period off the Client's monthly stream (ticket 12: a Decoy-described payment
    on an empty slot of a stream's schedule at its amount joins it); a Client with no stream gets them
    on fixed days."""
    rows = [json.loads(line) for line in (project.raw / SPLIT_FILES[split]).read_text().splitlines()]
    last = {}  # each Client's last stream payment
    for r in rows:
        if r["description"] != shop(r["client_id"])[0]["description"]:
            last[r["client_id"]] = max(pd.Timestamp(r["timestamp"]), last.get(r["client_id"], pd.Timestamp(r["timestamp"])))
    extra = []
    for client in sorted({r["client_id"] for r in rows}):
        for i, (description, mcc, amount) in enumerate(
            [("digital order", "5732", 6.8), ("merchant charge", "7997", 66.0), ("card purchase", "6300", 109.0)]
        ):
            start = last[client] - (15 + 30 * (i + 1)) * DAY if client in last else pd.Timestamp(f"2025-{8 + i:02d}-14")
            extra += series(client, description, mcc, amount, start, 3, every_days=-30 if client in last else 31)
        refund = series(client, "digital order", "7997", 66.0, "2025-12-20", 1)[0]
        extra.append({**refund, "type": "refund", "direction": "in"})
        extra.append({**shop(client, "2025-12-21")[0], "type": "refund", "direction": "in"})
    write_transactions(project, split, rows + extra)
    return data.load_transactions(project.raw, split)


def test_decoys_leave_the_none_models_training_rows_and_its_serving_input_unchanged(fetched, monkeypatch):
    separable_project(fetched)
    transactions, labels = train_split(fetched)
    valid = data.load_transactions(fetched.raw, "valid")
    clients = pd.Index(sorted(valid["client_id"].unique()), name="client_id")
    model = RankerModel(none_model=NoneModel()).fit(transactions, labels)

    served = []  # every input the none model is asked to score
    predict = NoneModel.predict

    def recording(self, x):
        served.append(x.copy())
        return predict(self, x)

    monkeypatch.setattr(NoneModel, "predict", recording)
    before_rows = model.none_training_features(transactions, labels)
    before_proba = model.predict_proba(valid, clients)

    decoyed, decoyed_valid = with_decoys(fetched, "train"), with_decoys(fetched, "valid")
    assert len(decoyed) > len(transactions) and len(decoyed_valid) > len(valid)
    pd.testing.assert_frame_equal(model.none_training_features(decoyed, labels), before_rows)
    after_proba = model.predict_proba(decoyed_valid, clients)
    assert len(served) == 2
    pd.testing.assert_frame_equal(served[1], served[0])
    pd.testing.assert_frame_equal(after_proba, before_proba)
    # a model fitted on the decoyed history is the same model
    refit = RankerModel(none_model=NoneModel()).fit(decoyed, labels)
    pd.testing.assert_frame_equal(refit.predict_proba(decoyed_valid, clients), before_proba)


def test_member_payments_are_the_streams_payments_with_the_ones_a_matched_refund_reverses():
    rows = stream_series("C1", "saas billing", "5734", 37.6, "2025-06-18", 6)
    rows += stream_series("C1", "gym membership", "7997", 66, "2025-04-10", 8)
    rows += [
        stream_tx("C1", rows[4]["timestamp"] + 3 * DAY, 37.9, "saas billing core", "5734", type_="refund", direction="in"),
        stream_tx("C1", "2025-07-01", 120, "hotel booking", "7011", type_="refund", direction="in"),
    ]
    rows += stream_series("C1", "neighborhood market", "5411", 40, "2025-07-06", 4)
    streams, payments = detect_stream_payments(frame(rows))

    pd.testing.assert_frame_equal(streams, detect_streams(frame(rows)))
    assert list(payments.columns) == ["client_id", "stream_id", "timestamp", "amount", "refunded"]
    counts = payments.groupby("stream_id").size()
    assert counts.to_dict() == streams.set_index("stream_id")["n_payments"].to_dict()
    software = streams.loc[streams["family"] == "software", "stream_id"].item()
    paid = payments[payments["stream_id"] == software]
    assert paid["timestamp"].is_monotonic_increasing
    assert paid["refunded"].tolist() == [False, False, False, False, True, False]
    assert not payments.loc[payments["stream_id"] != software, "refunded"].any()


def test_a_refund_that_fits_two_same_amount_payments_reverses_the_later_one():
    # a double charge: gym 66 on 11-02 and again on 11-05, refunded once on 11-07. Both payments are
    # inside the 7-day window and equally close in amount, so the latest one is reversed
    rows = stream_series("C1", "gym membership", "7997", 66, "2025-07-05", 6)  # 07-05 .. 12-02
    rows.append(stream_tx("C1", "2025-11-05", 66, "gym membership", "7997"))
    rows.append(stream_tx("C1", "2025-11-07", 66, "gym membership", "7997", type_="refund", direction="in"))
    rows.sort(key=lambda r: r["timestamp"])
    streams, payments = detect_stream_payments(frame(rows))
    assert (streams["n_payments"].item(), streams["n_refunds"].item()) == (7, 1)
    assert payments.loc[payments["refunded"], "timestamp"].tolist() == [pd.Timestamp("2025-11-05", tz="UTC")]

    # a second refund reverses the other one: each payment is reversed at most once
    rows.append(stream_tx("C1", "2025-11-08", 66, "gym membership", "7997", type_="refund", direction="in"))
    _, payments = detect_stream_payments(frame(rows))
    assert payments.loc[payments["refunded"], "timestamp"].tolist() == [
        pd.Timestamp("2025-11-02", tz="UTC"),
        pd.Timestamp("2025-11-05", tz="UTC"),
    ]


# --- the features, pinned on a hand-made history ---------------------------------------------------


def days_before_the_cutoff(days):
    return pd.Timestamp("2026-01-01", tz="UTC") - days * DAY


def pinned_history():
    """Client A: two Active Streams (gym with one missed payment and a refund of its last payment, and
    software), an ended insurance stream of exactly three payments and a short cloud stream. Client B:
    one mobile stream that ended long ago. Client C only shops. Times are days before the Cutoff."""

    def paid(client, days, amount, description, mcc, **kw):
        return [stream_tx(client, days_before_the_cutoff(d), amount, description, mcc, **kw) for d in days]

    rows = paid("A", [203, 174, 144, 83, 52, 20], 66, "gym membership", "7997")  # gaps 29 30 61 31 32
    rows += paid("A", [18], 66, "gym membership", "7997", type_="refund", direction="in")
    rows += paid("A", [115, 85, 55, 25], 39, "saas billing", "5734")  # gaps 30 30 30
    rows += paid("A", [152, 122, 92], 109, "insurance monthly", "6300")  # stopped 92 days ago
    rows += paid("A", [300, 270], 6.8, "cloud backup", "5732")  # two payments only
    rows += paid("A", [210, 150, 40], 42.1, "neighborhood market", "5411")  # shopping: no stream
    rows += paid("B", [290, 260, 230, 200], 47, "phone contract", "4814")  # stopped 200 days ago
    rows += paid("C", [100, 50], 42.1, "neighborhood market", "5411")
    return frame(rows)


def test_every_none_model_feature_matches_its_hand_computed_value():
    streams, payments = detect_stream_payments(pinned_history())
    detected = streams.set_index(["client_id", "family"])[["n_payments", "active", "n_refunds"]]
    assert detected.to_dict("index") == {
        ("A", "cloud"): {"n_payments": 2, "active": False, "n_refunds": 0},
        ("A", "gym"): {"n_payments": 6, "active": True, "n_refunds": 1},
        ("A", "insurance"): {"n_payments": 3, "active": False, "n_refunds": 0},
        ("A", "software"): {"n_payments": 4, "active": True, "n_refunds": 0},
        ("B", "mobile"): {"n_payments": 4, "active": False, "n_refunds": 0},
    }
    x = client_features(streams, payments)  # the default Cutoff: 2026-01-01
    assert list(x.index) == ["A", "B"]  # C has no stream
    assert list(x.columns) == FEATURE_COLUMNS

    # gym: gaps 29 30 61 31 32; the monthly base 30.4 folds 61 into two -> 29 30 30.5 31 32, median 30.5
    p = 30.5
    gym = {
        "n_payments": 6,
        "days_since_first": 203,
        "days_since_last": 20,
        "period_days": p,
        "gap_mad_days": 0.5,  # median of |folded - 30.5| = 1.5 0.5 0 0.5 1.5
        "amount_cv": 0.0,
        "amount": 66.0,
        "refund_rate": 1 / 6,
        "days_to_next": 10.5,  # last + 30.5: 20 days ago + 30.5 days, so ahead of the Cutoff
        "overdue": 20 / p,  # since the last payment, not the first
        "last_gap_ratio": 32 / p,
        "last_gap_missed": 0.0,
        "last_gap_late": (32 - p) / p,
        "n_missed": 1.0,  # folds 1 1 2 1 1: one missed payment
        "missed_rate": 1 / 6,  # missed / sum of folds
        "missed_last3": 1.0,  # folds 2 1 1
        "gap_trend": 0.7 / p,  # slope of 29 30 30.5 31 32 over 0..4 is 7/10: the gaps lengthen, so > 0
        "gap_recent_minus_all": ((30.5 + 31 + 32) / 3 - 30.5) / p,
        "gap_mad_rel": 0.5 / p,
        "expected_minus_actual": 183 / p + 1 - 6,  # 183 days span 6 periods: 7 payments due, 6 made
        "phase": 20 / p,
        "refund_last": 1.0,  # the refund 18 days ago reverses the payment 20 days ago
        "refund_last2": 1.0,
        "refund_last_days": 20.0,
        "n_refunds": 1.0,
        "last_dom": 12.0,  # 2025-12-12
        "first_dom": 12.0,  # 2025-06-12
        "next_dom": 11.0,  # 2026-01-11 12:00
    }
    software = {
        "n_payments": 4,
        "days_since_first": 115,
        "days_since_last": 25,
        "period_days": 30.0,
        "gap_mad_days": 0.0,
        "amount_cv": 0.0,
        "amount": 39.0,
        "refund_rate": 0.0,
        "days_to_next": 5.0,  # last + 30: 25 days ago + 30 days
        "overdue": 25 / 30,
        "last_gap_ratio": 1.0,
        "last_gap_missed": 0.0,
        "last_gap_late": 0.0,
        "n_missed": 0.0,
        "missed_rate": 0.0,
        "missed_last3": 0.0,
        "gap_trend": 0.0,
        "gap_recent_minus_all": 0.0,
        "gap_mad_rel": 0.0,
        "expected_minus_actual": 90 / 30 + 1 - 4,
        "phase": 25 / 30,
        "refund_last": 0.0,
        "refund_last2": 0.0,
        "refund_last_days": np.nan,  # never refunded
        "n_refunds": 0.0,
        "last_dom": 7.0,  # 2025-12-07
        "first_dom": 8.0,  # 2025-09-08
        "next_dom": 6.0,  # 2026-01-06
    }
    expected = {}
    for field in gym:
        # software is due first (5 days against 10.5): the soonest stream's value is its own, the
        # smaller of the two for n_payments (4 against 6) but the larger for days_since_last (25 against 20)
        expected[f"soonest_{field}"] = software[field]
        expected[f"min_{field}"] = np.nanmin([gym[field], software[field]])
        expected[f"max_{field}"] = np.nanmax([gym[field], software[field]])
    # the ended insurance stream (3 payments, the last 92 days ago) and the short cloud one (2 payments)
    # count in the stream totals, never in the Active Streams' values
    expected |= {
        "n_streams": 4,
        "n_active_streams": 2,
        "n_ended": 1,  # insurance: inactive, with exactly three payments
        "n_short": 1,  # cloud
        "past_churn_rate": 1 / (1 + 2),  # ended / (ended + active); the short stream is neither
        "n_ended_recent120": 1,  # insurance stopped 92 < 120 days ago
        "n_families_active": 2,
        "n_families_all": 4,
        "total_payments": 2 + 6 + 3 + 4,
        "stream_first_days": 300,  # the cloud stream's first payment, the earliest of all
        "stream_last_days": 20,
        # member payments (days ago): 300 270 | 203 174 144 83 52 20 | 152 122 92 | 115 85 55 25, span 300
        # 60 days: 52 20 55 25 -> 4 recent; 11 before, over (300 - 60) / 60 = 4 windows
        "stream_activity_ratio60": 4 / (11 / 4),
        # 90 days: 83 52 20 85 55 25 -> 6 recent; 9 before, over (300 - 90) / 90 windows
        "stream_activity_ratio90": 6 / (9 / (210 / 90)),
    }
    expected |= {f"active_has_{f}": float(f in ("gym", "software")) for f in LABELS if f != "none"}
    assert set(expected) == set(FEATURE_COLUMNS)
    assert x.loc["A"].to_dict() == pytest.approx(expected, rel=1e-12, abs=1e-12, nan_ok=True)
    assert x.loc["A", "max_gap_trend"] > 0 and x.loc["A", "soonest_days_to_next"] > 0

    # B: no Active Stream, so every value over Active Streams is unknown
    aggregated = [c for c in FEATURE_COLUMNS if c.startswith(("soonest_", "min_", "max_"))]
    has = [c for c in FEATURE_COLUMNS if c.startswith("active_has_")]
    b = x.loc["B"]
    assert b[aggregated].isna().all()
    assert (b[has] == 0).all()
    assert b.drop(aggregated + has).to_dict() == {
        "n_streams": 1,
        "n_active_streams": 0,
        "n_ended": 1,  # four payments, the last 200 days ago
        "n_short": 0,
        "past_churn_rate": 1.0,
        "n_ended_recent120": 0,  # 200 days ago
        "n_families_active": 0,
        "n_families_all": 1,
        "total_payments": 4,
        "stream_first_days": 290,
        "stream_last_days": 200,
        "stream_activity_ratio60": 0.0,  # nothing paid in the last 60 or 90 days of a 290-day span
        "stream_activity_ratio90": 0.0,
    }
