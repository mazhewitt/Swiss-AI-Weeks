"""The Stream Ranker (`--model ranker`) through the CLI against fixture data (Seam 1), plus its
per-Client probability rule on hand-set candidate scores."""

import json

import numpy as np
import pandas as pd
import pytest

from recurring_family.ranker import RankerModel, client_proba
from test_lgbm_pipeline import (
    FAMILY_STREAM,
    LABELS,
    SPLIT_FILES,
    read_rows,
    separable_split,
    series,
    shop,
    write_labels,
    write_sample_submission,
    write_transactions,
)


def read_proba(path):
    return pd.read_csv(path, dtype={"client_id": str}).set_index("client_id")


def evaluate_proba(project, *extra):
    out = project.root / "proba.csv"
    assert project.run("evaluate", "--model", "ranker", "--proba", str(out), *extra) == 0
    return read_proba(out)


def no_stream_clients(prefix, n):
    """Clients who only shop: they have no Candidate Streams at all."""
    clients = [f"{prefix}{i:03d}" for i in range(n)]
    return clients, [row for c in clients for row in shop(c)]


def separable_project(project, seed=3, per_label=20):
    rng = np.random.default_rng(seed)
    train = separable_split(project, "train", "T", per_label, rng)
    valid = separable_split(project, "valid", "V", 8, rng)
    return train, valid


# --- the probability table -------------------------------------------------------------


def test_ranker_returns_exactly_the_allowed_labels_one_row_per_requested_client_summing_to_one(fetched):
    separable_project(fetched)
    assert fetched.run("train", "--model", "ranker") == 0
    proba = evaluate_proba(fetched, "--change", "ranker on separable families")

    assert fetched.run("split") == 0
    split = pd.read_csv(fetched.root / "artifacts" / "valid_split.csv", dtype=str)
    assert list(proba.columns) == LABELS
    assert proba.index.is_unique
    assert set(proba.index) == set(split.query("set == 'selection'")["client_id"])
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all().all()

    row = read_rows(fetched.log)[-1]
    assert (row["model"], row["split"]) == ("ranker", "selection")
    assert float(row["macro_f1"]) >= 0.9, row  # every label is carried by an obvious stream


def test_a_client_without_candidate_streams_gets_a_none_dominated_row(fetched):
    separable_project(fetched)
    empty, rows = no_stream_clients("E", 12)
    valid = [json.loads(line) for line in (fetched.raw / SPLIT_FILES["valid"]).read_text().splitlines()]
    write_transactions(fetched, "valid", valid + rows)
    labels = pd.read_csv(fetched.raw / "valid_labels.csv", dtype=str)
    write_labels(
        fetched, "valid",
        dict(zip(labels["client_id"], labels["target_next_recurring_merchant"])) | {c: "none" for c in empty},
    )
    assert fetched.run("train", "--model", "ranker") == 0
    proba = evaluate_proba(fetched)

    scored = proba.loc[proba.index.intersection(empty)]
    assert len(scored)  # some of them fall in the selection set
    assert (scored.idxmax(axis=1) == "none").all()
    assert (scored["none"] > 0.5).all()
    assert np.allclose(scored.sum(axis=1), 1.0)


def test_client_proba_takes_each_familys_best_candidate_and_none_as_one_minus_the_best():
    candidates = pd.DataFrame(
        {
            "client_id": ["A", "A", "A", "B"],
            "family": ["gym", "gym", "cloud", "music"],
            "score": [0.3, 0.6, 0.2, 0.1],
        }
    )
    proba = client_proba(candidates, pd.Index(["A", "B", "C"]))
    assert list(proba.columns) == LABELS
    assert proba.index.tolist() == ["A", "B", "C"]
    assert np.allclose(proba.sum(axis=1), 1.0)
    # A: gym 0.6 (its better stream), cloud 0.2, none 1 - 0.6, normalised by 1.2
    a = proba.loc["A"]
    assert a["gym"] == pytest.approx(0.6 / 1.2)
    assert a["cloud"] == pytest.approx(0.2 / 1.2)
    assert a["none"] == pytest.approx(0.4 / 1.2)
    assert a[["insurance", "mobile", "music", "software", "streaming"]].sum() == 0
    # B: one weak candidate
    assert proba.loc["B", "music"] == pytest.approx(0.1 / 1.0)
    assert proba.loc["B", "none"] == pytest.approx(0.9)
    # C: no candidates at all
    assert proba.loc["C", "none"] == 1.0


def test_two_streams_in_the_same_family_both_feed_that_familys_probability():
    # whichever of a Client's two gym streams scores higher sets the gym probability,
    # and the pair never splits it: gym gets the same share as with the better stream alone
    both = pd.DataFrame({"client_id": ["A", "A"], "family": ["gym", "gym"], "score": [0.7, 0.2]})
    swapped = both.assign(score=[0.2, 0.7])
    alone = both.iloc[[0]]
    clients = pd.Index(["A"])
    for table in (swapped, alone):
        pd.testing.assert_frame_equal(client_proba(table, clients), client_proba(both, clients))
    assert client_proba(both, clients).loc["A", "gym"] == pytest.approx(0.7)


def test_a_client_with_two_streams_in_one_family_gets_one_row_led_by_that_family(fetched):
    separable_project(fetched)
    # D pays two gym memberships at different prices: two gym Candidate Streams
    rows = series("D", "gym membership", "7997", 66, "2025-07-10", 6)
    rows += series("D", "fitness club", "7997", 29, "2025-08-20", 5)
    write_transactions(fetched, "test", rows)
    write_sample_submission(fetched, ["D"])
    assert fetched.run("train", "--model", "ranker") == 0
    assert fetched.run("submit", "--model", "ranker", "--name", "two-gym") == 0
    sub = pd.read_csv(fetched.root / "submissions" / "two-gym.csv", dtype=str)
    assert sub.to_dict("records") == [{"client_id": "D", "predicted_next_recurring_merchant": "gym"}]


# --- cross-validation, the decision layer, persistence and submission ---------------------


def test_cv_folds_by_client_giving_one_out_of_fold_row_per_labelled_client(fetched):
    train, _ = separable_project(fetched)
    out = fetched.root / "oof.csv"
    assert fetched.run("cv", "--model", "ranker", "--out", str(out)) == 0
    oof = pd.read_csv(out, dtype={"client_id": str})
    assert list(oof.columns) == ["client_id", "fold", *LABELS]
    assert sorted(oof["client_id"]) == sorted(train)  # one row each: no Client straddles folds
    assert sorted(oof["fold"].unique()) == [0, 1, 2, 3, 4]
    assert np.allclose(oof[LABELS].sum(axis=1), 1.0)
    predicted = oof.set_index("client_id")[LABELS].idxmax(axis=1)
    assert (predicted == pd.Series(train)).mean() >= 0.9


def test_train_with_the_tuned_decision_fits_it_on_out_of_fold_probabilities(fetched, capsys):
    train, _ = separable_project(fetched)
    assert fetched.run("train", "--model", "ranker", "--decision", "tuned") == 0
    assert "decision layer tuned on 5-fold out-of-fold probabilities" in capsys.readouterr().out
    decision = json.loads((fetched.root / "artifacts" / "ranker.decision.json").read_text())
    assert sorted(decision["fitted_on_clients"]) == sorted(train)
    assert set(decision["weights"]) == set(LABELS)

    assert fetched.run("evaluate", "--model", "ranker", "--decision", "tuned") == 0
    row = read_rows(fetched.log)[-1]
    assert (row["model"], row["split"]) == ("ranker+tuned", "selection")


def test_save_then_load_gives_identical_probabilities(fetched, tmp_path):
    separable_project(fetched)
    from recurring_family import data

    transactions = data.load_transactions(fetched.raw, "train")
    labels = data.load_labels(fetched.raw, "train").set_index("client_id")["target_next_recurring_merchant"]
    valid = data.load_transactions(fetched.raw, "valid")
    clients = pd.Index(sorted(valid["client_id"].unique()), name="client_id")

    model = RankerModel().fit(transactions, labels)
    before = model.predict_proba(valid, clients)
    model.save(tmp_path / "ranker.json")
    after = RankerModel.load(tmp_path / "ranker.json").predict_proba(valid, clients)
    pd.testing.assert_frame_equal(after, before, check_exact=True)
    assert before["none"].min() < 0.5  # a real model, not a constant


def test_submit_with_the_ranker_writes_a_valid_submission(fetched):
    separable_project(fetched)
    tx = [json.loads(line) for line in (fetched.raw / SPLIT_FILES["test"]).read_text().splitlines()]
    write_transactions(fetched, "test", tx + shop("C000099"))
    write_sample_submission(fetched, ["C000004", "C000008", "C000010", "C000099"])
    assert fetched.run("train", "--model", "ranker", "--decision", "tuned") == 0
    assert fetched.run("submit", "--model", "ranker", "--decision", "tuned", "--name", "ranker") == 0
    path = fetched.root / "submissions" / "ranker.csv"
    sub = pd.read_csv(path, dtype=str)
    assert sorted(sub["client_id"]) == ["C000004", "C000008", "C000010", "C000099"]
    assert set(sub["predicted_next_recurring_merchant"]) <= set(LABELS)
    assert sub.set_index("client_id").loc["C000099", "predicted_next_recurring_merchant"] == "none"
    assert fetched.run("submit", "--check", str(path)) == 0


def test_the_ranker_fits_on_train_even_when_every_candidate_is_negative(fetched):
    # every labelled Client is none: no candidate carries a label, so there is nothing to rank
    rows = series("N1", "gym membership", "7997", 66, "2025-02-01", 5) + shop("N2")
    write_transactions(fetched, "train", rows)
    write_labels(fetched, "train", {"N1": "none", "N2": "none"})
    assert fetched.run("train", "--model", "ranker") == 0
    proba = evaluate_proba(fetched)
    assert (proba["none"] == 1.0).all()


# --- ADR 0001: stream features only -------------------------------------------------------


def valid_rows(project):
    return [json.loads(line) for line in (project.raw / SPLIT_FILES["valid"]).read_text().splitlines()]


def test_decoys_and_description_wording_never_change_the_ranker_probabilities(fetched):
    separable_project(fetched)
    assert fetched.run("train", "--model", "ranker") == 0
    before = evaluate_proba(fetched)

    rows = valid_rows(fetched)
    # Filler Descriptions inside real streams change only the stream's description statistics
    for row in rows:
        if row["description"] == FAMILY_STREAM["gym"][0] and row["timestamp"] >= "2025-11-01":
            row["description"] = "member plan"
    # Decoy Transactions on stream MCCs and at stream amounts
    decoys = []
    for client in sorted({r["client_id"] for r in rows}):
        for i, (description, mcc, amount) in enumerate(
            [("digital order", "5732", 6.8), ("merchant charge", "7997", 66.0), ("card purchase", "6300", 109.0)]
        ):
            decoys += series(client, description, mcc, amount, f"2025-{8 + i:02d}-14", 3, every_days=31)
    write_transactions(fetched, "valid", rows + decoys)
    after = evaluate_proba(fetched)
    pd.testing.assert_frame_equal(after, before)
