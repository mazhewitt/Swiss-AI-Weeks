"""The Stream Ranker with its Client-level `none` model (`--none-model`): through the CLI against fixture
data (Seam 1), the stream module's member payments (Seam 2), and the combining rule on hand-set inputs."""

import json

import numpy as np
import pandas as pd
import pytest

from recurring_family import data
from recurring_family.none_model import FEATURE_COLUMNS, RANKER_NONE, NoneModel, client_features, combine
from recurring_family.ranker import RankerModel
from recurring_family.streams import detect_stream_payments, detect_streams
from test_lgbm_pipeline import (
    FAMILY_STREAM,
    LABELS,
    SPLIT_FILES,
    read_rows,
    separable_split,
    series,
    shop,
    write_sample_submission,
    write_transactions,
)
from test_ranker import evaluate_proba, valid_rows
from test_ranker_pseudo_labels import pooled_project
from test_streams import frame
from test_streams import series as stream_series
from test_streams import tx as stream_tx

DAY = pd.Timedelta(days=1)
LABEL_COLUMN = "target_next_recurring_merchant"


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
