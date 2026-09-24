"""Seam 1: E2 (stream features + LightGBM) through the CLI against fixture data."""

import csv
import json

import numpy as np
import pandas as pd
import pytest

LABELS = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none"]
FAMILIES = LABELS[:-1]
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
SPLIT_FILES = {
    "train": "train_transactions.jsonl",
    "valid": "valid_transactions.jsonl",
    "test": "test_transactions.jsonl",
    "unlabeled": "unlabeled_pretrain_transactions.jsonl",
}

# One easily recognised stream per family: (description, home MCC, amount).
FAMILY_STREAM = {
    "cloud": ("cloud backup", "5732", 6.8),
    "gym": ("gym membership", "7997", 66.0),
    "insurance": ("cover plan", "6300", 109.0),
    "mobile": ("phone contract", "4814", 47.0),
    "music": ("audio streaming", "5812", 13.5),
    "software": ("saas billing", "5734", 39.0),
    "streaming": ("media streaming", "5812", 17.9),
}


# --- fixture builders -----------------------------------------------------------


def tx(client, when, amount, description, mcc, type_="card_payment", direction="out"):
    return {
        "amount": float(amount),
        "client_id": client,
        "currency": "chf",
        "description": description,
        "direction": direction,
        "fee": 0.0,
        "mcc": mcc,
        "timestamp": pd.Timestamp(when).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "type": type_,
    }


def series(client, description, mcc, amount, start, n, every_days=30):
    first = pd.Timestamp(start)
    return [tx(client, first + pd.Timedelta(days=i * every_days), amount, description, mcc) for i in range(n)]


def shop(client, when="2025-10-11"):
    return [tx(client, when, 42.1, "neighborhood market", "5411")]


def write_transactions(project, split, rows):
    with open(project.raw / SPLIT_FILES[split], "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_labels(project, split, labels):
    pd.DataFrame(
        {"client_id": list(labels), "cutoff_date": "2026-01-01", "target_next_recurring_merchant": list(labels.values())}
    ).to_csv(project.raw / f"{split}_labels.csv", index=False)


def write_sample_submission(project, clients):
    pd.DataFrame({"client_id": clients, "predicted_next_recurring_merchant": "none"}).to_csv(
        project.raw / "sample_submission.csv", index=False
    )


def features(project, split):
    out = project.root / f"features-{split}.csv"
    assert project.run("features", "--split", split, "--out", str(out)) == 0
    header = pd.read_csv(out, nrows=0).columns
    dtypes = {c: str for c in header if c == "client_id" or c.endswith(("_family", "_mcc"))}
    return pd.read_csv(out, dtype=dtypes).set_index("client_id")


def days_before_cutoff(when):
    return (CUTOFF - pd.Timestamp(when, tz="UTC")) / pd.Timedelta(days=1)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --- one feature row per Client, from the stream table only ------------------------


def test_train_lgbm_builds_one_feature_row_per_client_including_clients_without_streams(fetched):
    assert fetched.run("train", "--model", "lgbm") == 0
    rows = features(fetched, "train")
    # the train fixture: C000001 gym, C000002 cloud, C000007 insurance (4 monthly payments each);
    # C000003, C000005 and C000009 only shop, so they have no Recurring Streams at all.
    assert rows.index.tolist() == ["C000001", "C000002", "C000003", "C000005", "C000007", "C000009"]
    gym = rows.loc["C000001"]
    assert (gym["slot1_family"], gym["slot1_mcc"]) == ("gym", "7997")
    assert gym["slot1_amount"] == pytest.approx(66.0)
    assert gym["slot1_period_days"] == pytest.approx(30.0)  # gaps 30, 31, 30
    assert gym["slot1_days_to_next"] == pytest.approx(3 + 10 / 24)  # 2025-12-05T10:00 + 30 days
    assert gym["slot1_n_payments"] == 4
    assert pd.isna(gym["slot2_family"])
    assert gym["n_active_streams"] == 1
    for client in ["C000003", "C000005", "C000009"]:
        empty = rows.loc[client]
        assert empty["n_active_streams"] == 0 and empty["longest_active_payments"] == 0
        assert all(pd.isna(empty[f"slot{k}_family"]) for k in (1, 2, 3))
        assert all(empty[f"{f}_active_streams"] == 0 for f in FAMILIES)


def test_lgbm_predicts_for_every_client_including_clients_without_streams(fetched):
    rows = [json.loads(line) for line in (fetched.raw / SPLIT_FILES["test"]).read_text().splitlines()]
    write_transactions(fetched, "test", rows + shop("C000099"))
    write_sample_submission(fetched, ["C000004", "C000008", "C000010", "C000099"])

    assert fetched.run("train", "--model", "lgbm") == 0
    proba_path = fetched.root / "proba.csv"
    assert fetched.run("evaluate", "--model", "lgbm", "--proba", str(proba_path)) == 0
    proba = pd.read_csv(proba_path, dtype={"client_id": str}).set_index("client_id")
    assert list(proba.columns) == LABELS
    assert proba.index.tolist() == ["C000011", "C000013", "C000014"]  # the selection set
    assert np.allclose(proba.sum(axis=1), 1.0)

    assert fetched.run("submit", "--model", "lgbm", "--name", "e2") == 0
    sub = pd.read_csv(fetched.root / "submissions" / "e2.csv", dtype=str)
    assert sorted(sub["client_id"]) == ["C000004", "C000008", "C000010", "C000099"]
    assert set(sub["predicted_next_recurring_merchant"]) <= set(LABELS)
    assert fetched.run("submit", "--check", str(fetched.root / "submissions" / "e2.csv")) == 0


def test_features_without_a_trained_lgbm_model_fail(fetched, capsys):
    assert fetched.run("features", "--split", "train") != 0
    assert "train --model lgbm" in capsys.readouterr().err


def slot_history():
    """Client A: four Active Streams, a stopped stream and a 2-payment stream."""
    rows = []
    rows += series("A", "gym membership", "7997", 66, "2025-07-13", 6)  # last 12-10, next 2026-01-09
    # cloud: gaps 30, 31, 29, 30 and amounts 6.8 / 7.0 alternating; last 12-05, next 2026-01-04
    for when, amount in [("2025-08-07", 6.8), ("2025-09-06", 7.0), ("2025-10-07", 6.8), ("2025-11-05", 7.0), ("2025-12-05", 6.8)]:
        rows.append(tx("A", when, amount, "cloud backup", "5732"))
    rows += series("A", "cover plan", "6300", 109, "2025-09-16", 4)  # last 12-15, next 2026-01-14
    rows += series("A", "phone contract", "4814", 47, "2025-10-21", 3)  # last 12-20, next 2026-01-19
    # stopped software stream (7 payments, last 2025-07-06): rolled forward it is due 2026-01-02,
    # sooner than any Active Stream, so it would take slot 1 if stopped streams were not excluded
    rows += series("A", "saas billing", "5734", 39, "2025-01-07", 7)
    # a 2-payment music stream due exactly at the Cutoff: never Active, never a slot
    rows += series("A", "audio streaming", "5812", 13.5, "2025-11-02", 2)
    rows += shop("A")
    return rows


def test_the_top_three_active_streams_fill_the_slots_soonest_next_payment_first(fetched):
    write_transactions(fetched, "valid", slot_history())
    assert fetched.run("train", "--model", "lgbm") == 0
    a = features(fetched, "valid").loc["A"]

    assert [a[f"slot{k}_family"] for k in (1, 2, 3)] == ["cloud", "gym", "insurance"]
    assert [a[f"slot{k}_mcc"] for k in (1, 2, 3)] == ["5732", "7997", "6300"]
    assert [a[f"slot{k}_days_to_next"] for k in (1, 2, 3)] == pytest.approx([3, 8, 13])
    assert [a[f"slot{k}_amount"] for k in (1, 2, 3)] == pytest.approx([6.8, 66, 109])
    assert [a[f"slot{k}_n_payments"] for k in (1, 2, 3)] == [5, 6, 4]
    assert [a[f"slot{k}_period_days"] for k in (1, 2, 3)] == pytest.approx([30, 30, 30])
    # regularity: the cloud stream's gap spread and amount variation; the others are exact
    amounts = np.array([6.8, 7.0, 6.8, 7.0, 6.8])
    assert a["slot1_gap_mad_days"] == pytest.approx(0.5)
    assert a["slot1_amount_cv"] == pytest.approx(amounts.std(ddof=1) / amounts.mean())
    assert (a["slot2_gap_mad_days"], a["slot2_amount_cv"]) == (0, 0)


def test_the_per_family_block_uses_active_and_three_plus_payment_streams_only(fetched):
    write_transactions(fetched, "valid", slot_history())
    assert fetched.run("train", "--model", "lgbm") == 0
    a = features(fetched, "valid").loc["A"]

    assert {f: a[f"{f}_active_streams"] for f in FAMILIES} == {
        "cloud": 1, "gym": 1, "insurance": 1, "mobile": 1, "music": 0, "software": 0, "streaming": 0,
    }
    assert a["gym_max_payments"] == 6
    assert a["gym_days_to_next"] == pytest.approx(8)
    assert a["gym_days_since_last"] == pytest.approx(days_before_cutoff("2025-12-10"))
    assert a["gym_days_since_first"] == pytest.approx(days_before_cutoff("2025-07-13"))
    # the stopped software stream still counts (7 payments) but has no next payment to wait for
    assert a["software_max_payments"] == 7
    assert pd.isna(a["software_days_to_next"])
    assert a["software_days_since_last"] == pytest.approx(days_before_cutoff("2025-07-06"))
    assert a["software_days_since_first"] == pytest.approx(days_before_cutoff("2025-01-07"))
    # the 2-payment music stream is left out, like a family with no stream at all
    for family in ("music", "streaming"):
        assert a[f"{family}_max_payments"] == 0
        assert pd.isna(a[f"{family}_days_since_last"]) and pd.isna(a[f"{family}_days_since_first"])


def test_the_none_signals_come_from_active_streams(fetched):
    rows = slot_history()
    # B: an Active gym stream with two Filler Descriptions, an Active "premium plan" streaming stream
    # and a stopped stream whose descriptions are all family-specific
    rows += series("B", "gym membership", "7997", 66, "2025-07-11", 4)
    rows += series("B", "member plan", "7997", 66, "2025-11-08", 2)
    rows += series("B", "premium plan", "5812", 18, "2025-09-12", 4)
    rows += series("B", "saas billing", "5734", 39, "2025-01-07", 5)
    write_transactions(fetched, "valid", rows)
    assert fetched.run("train", "--model", "lgbm") == 0
    rows = features(fetched, "valid")

    a, b = rows.loc["A"], rows.loc["B"]
    assert (a["n_active_streams"], a["n_streams_3plus"]) == (4, 5)
    assert a["longest_active_payments"] == 6  # the stopped software stream has 7
    assert a["earliest_active_start_days"] == pytest.approx(days_before_cutoff("2025-07-13"))  # not 2025-01-07
    assert a["family_description_share"] == pytest.approx(1.0)
    # B's Active Streams: 4 of 6 gym payments name the family, none of the 4 "premium plan" ones do
    assert (b["n_active_streams"], b["longest_active_payments"]) == (2, 6)
    assert b["earliest_active_start_days"] == pytest.approx(days_before_cutoff("2025-07-11"))
    assert b["family_description_share"] == pytest.approx(4 / 10)


def test_a_stream_paid_mostly_on_an_unexpected_mcc_encodes_its_mcc_as_unknown(fetched):
    rows = series("C", "cloud backup", "5732", 6.8, "2025-06-05", 3)
    rows += series("C", "cloud backup", "5999", 6.8, "2025-09-03", 4)
    write_transactions(fetched, "valid", rows)
    assert fetched.run("train", "--model", "lgbm") == 0
    c = features(fetched, "valid").loc["C"]
    assert (c["slot1_family"], c["slot1_mcc"]) == ("cloud", "unknown")


def test_decoys_and_other_raw_transactions_never_change_a_feature_row(fetched):
    # ADR 0001: every feature comes from the stream table, so raw transaction counts, spend and
    # Decoy Transactions (on stream MCCs and at stream amounts too) leave the rows unchanged.
    assert fetched.run("train", "--model", "lgbm") == 0
    before = features(fetched, "valid")
    path = fetched.raw / SPLIT_FILES["valid"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    extra = []
    for client in ["C000011", "C000012", "C000013", "C000014"]:
        for i, (description, mcc, amount) in enumerate(
            [("digital order", "5732", 6.8), ("merchant charge", "7997", 66.0), ("service payment", "5812", 4.0),
             ("card purchase", "6300", 67.0), ("digital order", "5999", 12.0), ("merchant charge", "4814", 47.0)]
        ):
            extra += series(client, description, mcc, amount, f"2025-{7 + i:02d}-14", 3, every_days=31)
        extra += series(client, "coffee shop", "5812", 4.5, "2025-06-01", 12, every_days=14)
    write_transactions(fetched, "valid", rows + extra)
    after = features(fetched, "valid")
    pd.testing.assert_frame_equal(after, before)


# --- one schema for every split ------------------------------------------------------


def test_the_feature_schema_is_identical_across_train_valid_test_and_unlabeled(fetched):
    # the splits hold different families (train: gym, cloud, insurance; test: streaming, mobile, ...)
    assert fetched.run("train", "--model", "lgbm") == 0
    headers = {}
    for split in SPLIT_FILES:
        rows = features(fetched, split)
        clients = {json.loads(line)["client_id"] for line in (fetched.raw / SPLIT_FILES[split]).read_text().splitlines()}
        assert sorted(rows.index) == sorted(clients), split
        headers[split] = list(rows.columns)
    assert headers["valid"] == headers["train"]
    assert headers["test"] == headers["train"]
    assert headers["unlabeled"] == headers["train"]

    columns = set(headers["train"])
    for k in (1, 2, 3):
        assert {f"slot{k}_{f}" for f in ("family", "mcc", "amount", "period_days", "gap_mad_days", "days_to_next")} <= columns
    for family in FAMILIES:
        assert {f"{family}_active_streams", f"{family}_max_payments", f"{family}_days_to_next"} <= columns
    assert {"longest_active_payments", "earliest_active_start_days", "family_description_share"} <= columns


def test_unseen_descriptions_encode_as_unknown(fetched):
    # Train: four "gym membership" streams whose Client's next family is gym, four "urban gym"
    # streams whose Client's label is none. So "gym membership" rates above the prior (4 of 8),
    # "urban gym" below it, and any description never seen in training gets the prior itself.
    rows, labels = [], {}
    for i in range(4):
        rows += series(f"G{i}", "gym membership", "7997", 66, "2025-07-10", 6)
        rows += series(f"N{i}", "urban gym", "7997", 66, "2025-07-10", 6)
        labels[f"G{i}"], labels[f"N{i}"] = "gym", "none"
    write_transactions(fetched, "train", rows)
    write_labels(fetched, "train", labels)
    test = series("T1", "gym membership", "7997", 66, "2025-07-10", 6)
    test += series("T2", "urban gym", "7997", 66, "2025-07-10", 6)
    test += series("T3", "gym membrsh", "7997", 66, "2025-07-10", 6)  # a truncation seen only in test
    test += series("T4", "billing fit club core", "7997", 66, "2025-07-10", 6)
    write_transactions(fetched, "test", test)
    write_sample_submission(fetched, ["T1", "T2", "T3", "T4"])

    assert fetched.run("train", "--model", "lgbm") == 0
    rate = features(fetched, "test")["slot1_description_rate"]
    assert rate["T3"] == pytest.approx(0.5) and rate["T4"] == pytest.approx(0.5)
    assert rate["T1"] > 0.5 > rate["T2"]
    assert fetched.run("submit", "--model", "lgbm", "--name", "unseen") == 0


# --- the model -----------------------------------------------------------------------


def no_stream_clients(project, n_none, n_gym):
    """Clients whose histories are identical (one shop payment, no streams): nothing separates them."""
    labels = {f"S{i:03d}": "none" for i in range(n_none)} | {f"S{100 + i:03d}": "gym" for i in range(n_gym)}
    write_transactions(project, "train", [row for c in labels for row in shop(c)])
    write_labels(project, "train", labels)
    return labels


def test_lgbm_uses_balanced_class_weights_and_saves_out_of_fold_probabilities_via_the_cv_harness(fetched):
    # 40 none and 20 gym Clients that no feature can tell apart: with balanced class weights the
    # model gives both classes the same probability (unweighted it would be 2/3 none).
    labels = no_stream_clients(fetched, 40, 20)
    out = fetched.root / "oof.csv"
    assert fetched.run("cv", "--model", "lgbm", "--out", str(out)) == 0
    oof = pd.read_csv(out, dtype={"client_id": str})
    assert list(oof.columns) == ["client_id", "fold", *LABELS]
    assert sorted(oof["client_id"]) == sorted(labels)
    assert sorted(oof["fold"].unique()) == [0, 1, 2, 3, 4]
    assert np.allclose(oof["none"], 0.5, atol=1e-3)
    assert np.allclose(oof["gym"], 0.5, atol=1e-3)
    assert (oof[[l for l in LABELS if l not in ("none", "gym")]] == 0).all().all()


def separable_history(client, label, rng):
    """An Active Stream of the labelled family, else (for none) no Active Stream at all."""
    rows = shop(client, f"2025-{rng.integers(6, 12):02d}-{rng.integers(1, 28):02d}")
    if label == "none":
        if rng.random() < 0.5:  # a stream that stopped in the summer
            family = FAMILIES[rng.integers(len(FAMILIES))]
            description, mcc, amount = FAMILY_STREAM[family]
            rows += series(client, description, mcc, amount, "2025-02-01", 5)
        return rows
    description, mcc, amount = FAMILY_STREAM[label]
    n = int(rng.integers(3, 10))
    last = pd.Timestamp("2025-12-01") + pd.Timedelta(days=int(rng.integers(0, 29)))
    return rows + series(client, description, mcc, amount * rng.uniform(0.97, 1.03), last - pd.Timedelta(days=30 * (n - 1)), n)


def separable_split(project, split, prefix, per_label, rng):
    rows, labels = [], {}
    for label in LABELS:
        for i in range(per_label):
            client = f"{prefix}{label[:3].upper()}{i:03d}"
            labels[client] = label
            rows += separable_history(client, label, rng)
    write_transactions(project, split, rows)
    write_labels(project, split, labels)
    return labels


def test_lgbm_reaches_macro_f1_of_at_least_095_on_easily_separable_families(fetched):
    rng = np.random.default_rng(6)
    separable_split(fetched, "train", "T", 25, rng)
    separable_split(fetched, "valid", "V", 10, rng)
    assert fetched.run("train", "--model", "lgbm") == 0
    assert fetched.run("evaluate", "--model", "lgbm", "--change", "E2 on separable families") == 0
    row = read_rows(fetched.log)[-1]
    assert row["model"] == "lgbm" and row["split"] == "selection"
    assert float(row["macro_f1"]) >= 0.95, row


def test_leakage_trap_label_derived_description_rates_are_fitted_out_of_fold(fetched):
    # Every Client has one Active gym stream with a description no one else has. The real signal is
    # the stream's shape: gym Clients have long streams that started early, none Clients short late
    # ones; 15% of labels disagree with it. A description rate fitted on the rows it is used for
    # would encode each training Client's own label, the model would trust it over the real signal,
    # and out of fold (where every description is unseen) it would be no better than a coin.
    rng = np.random.default_rng(7)
    rows, labels = [], {}
    for i in range(200):
        client = f"L{i:03d}"
        long_stream = i % 2 == 0
        start, n = ("2025-05-14", 8) if long_stream else ("2025-10-12", 3)
        rows += series(client, f"gym membership x{i:03d}", "7997", 66, start, n)
        truth = "gym" if long_stream else "none"
        labels[client] = truth if rng.random() >= 0.15 else ("none" if truth == "gym" else "gym")
    write_transactions(fetched, "train", rows)
    write_labels(fetched, "train", labels)

    out = fetched.root / "oof.csv"
    assert fetched.run("cv", "--model", "lgbm", "--out", str(out)) == 0
    oof = pd.read_csv(out, dtype={"client_id": str}).set_index("client_id")
    predicted = oof[LABELS].idxmax(axis=1)
    accuracy = (predicted == pd.Series(labels)).mean()
    assert accuracy >= 0.75, accuracy
