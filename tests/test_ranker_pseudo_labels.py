"""The Stream Ranker trained with Pseudo-Labels (`--pseudo`) through the CLI against fixture data (Seam 1).

Pseudo-Labelled Clients come from any split at a Shifted Cutoff: their Candidate Streams are built
from the transactions before it and their labels from what they paid next. They join the fit as
weighted rows, but never a validation fold or the decision layer's fit.
"""

import json

import numpy as np
import pandas as pd
import pytest

from recurring_family.evaluation import append_log
from test_lgbm_pipeline import (
    FAMILIES,
    FAMILY_STREAM,
    LABELS,
    read_rows,
    separable_split,
    series,
    shop,
    write_labels,
    write_transactions,
)

SHIFTED = "2025-10-03"


def pseudo_histories(project, split, prefix, per_label=12):
    """Clients with no label file whose Pseudo-Labels at 2025-10-03 are obvious.

    A family Client pays monthly from some month between February and July to December 2025: at the
    Shifted Cutoff it has an Active Stream of three to eight payments and pays it again inside the
    Horizon. A `none` Client (three times as many) has a stream that stopped by June, so nothing
    recurs in the Horizon. Returns {client: its Pseudo-Label}.
    """
    rows, labels = [], {}
    for label in LABELS:
        for i in range(per_label * (3 if label == "none" else 1)):
            client = f"{prefix}{label[:3].upper()}{i:03d}"
            labels[client] = label
            rows += shop(client, "2025-03-11")
            family = label if label != "none" else FAMILIES[i % len(FAMILIES)]
            description, mcc, amount = FAMILY_STREAM[family]
            first_month = 2 + i % 6
            n = 12 - first_month if label != "none" else 4
            rows += series(client, description, mcc, amount, f"2025-{first_month:02d}-{1 + i % 25:02d}", n)
    write_transactions(project, split, rows)
    return labels


def pooled_project(project, seed=3):
    rng = np.random.default_rng(seed)
    train = separable_split(project, "train", "T", 12, rng)
    separable_split(project, "valid", "V", 8, rng)
    unlabeled = pseudo_histories(project, "unlabeled", "U")
    return train, unlabeled


def meta(project):
    return json.loads((project.root / "artifacts" / "ranker.meta.json").read_text())


def add_fidelity_row(project, verdict, run_id):
    scores = {"macro_f1": 0.6, **{f"f1_{label}": 0.6 for label in LABELS}}
    append_log(
        project.log,
        {"row_type": "fidelity", "run_id": run_id, "model": "rules+gate4", "split": "train",
         "verdict": verdict, **{k: f"{v:.4f}" for k, v in scores.items()}},
    )


# --- Pseudo-Labelled Clients are training rows ----------------------------------------------


def test_pseudo_labelled_clients_teach_a_ranker_whose_real_labels_are_all_none(fetched):
    # every real train label is none, so on its own the ranker can only ever say none;
    # the unlabelled Clients' Pseudo-Labels are what teach it which stream recurs
    write_transactions(fetched, "train", [row for c in ("N1", "N2", "N3") for row in shop(c)])
    write_labels(fetched, "train", {"N1": "none", "N2": "none", "N3": "none"})
    separable_split(fetched, "valid", "V", 8, np.random.default_rng(4))
    pseudo_histories(fetched, "unlabeled", "U")

    assert fetched.run("train", "--model", "ranker") == 0
    assert fetched.run("evaluate", "--model", "ranker", "--change", "real labels only") == 0
    real_only = read_rows(fetched.log)[-1]

    assert fetched.run("train", "--model", "ranker", "--pseudo", "unlabeled") == 0
    assert fetched.run("evaluate", "--model", "ranker", "--change", "with Pseudo-Labels") == 0
    pooled = read_rows(fetched.log)[-1]

    assert float(real_only["macro_f1"]) < 0.2, real_only  # none only
    assert float(pooled["macro_f1"]) >= 0.9, pooled
    assert pooled["model"] == "ranker+pseudo"


def test_train_records_the_sources_shifted_cutoffs_weight_and_clients_in_the_model_metadata(fetched, capsys):
    train, unlabeled = pooled_project(fetched)
    assert fetched.run(
        "train", "--model", "ranker", "--pseudo", "unlabeled", "--pseudo", "train:2025-09-01",
        "--pseudo-weight", "0.25",
    ) == 0
    out = capsys.readouterr().out
    saved = meta(fetched)
    assert saved["fitted_on"] == ["train"]
    assert saved["pseudo"]["sources"] == [
        {"split": "unlabeled", "cutoff": SHIFTED, "clients": len(unlabeled)},
        {"split": "train", "cutoff": "2025-09-01", "clients": len(train)},
    ]
    assert saved["pseudo"]["weight"] == 0.25
    assert saved["pseudo"]["min_payments"] == 4
    assert saved["training_clients"] == 2 * len(train) + len(unlabeled)
    # a train Client counts twice: with its real label and as a Pseudo-Labelled Client
    assert f"{len(train)} real-labelled + {len(train) + len(unlabeled)} Pseudo-Labelled Clients" in out
    assert "unlabeled@2025-10-03" in out and "train@2025-09-01" in out and "weight 0.25" in out


def test_the_default_weight_is_one_half_and_real_label_training_records_no_pseudo_labels(fetched):
    train, unlabeled = pooled_project(fetched)
    assert fetched.run("train", "--model", "ranker", "--pseudo", "unlabeled") == 0
    assert meta(fetched)["pseudo"]["weight"] == 0.5

    assert fetched.run("train", "--model", "ranker") == 0
    assert meta(fetched)["pseudo"] is None
    assert meta(fetched)["training_clients"] == len(train)


def test_the_pseudo_label_weight_changes_the_fitted_model(fetched):
    pooled_project(fetched)
    probas = []
    for weight in ("0.1", "2"):
        out = fetched.root / f"proba-{weight}.csv"
        assert fetched.run("train", "--model", "ranker", "--pseudo", "unlabeled", "--pseudo-weight", weight) == 0
        assert fetched.run("evaluate", "--model", "ranker", "--proba", str(out)) == 0
        probas.append(pd.read_csv(out, dtype={"client_id": str}).set_index("client_id"))
    assert not np.allclose(probas[0].to_numpy(), probas[1].to_numpy())


# --- pseudo rows never validate -------------------------------------------------------------


def test_cv_out_of_fold_output_contains_only_the_real_labelled_clients(fetched, capsys):
    train, unlabeled = pooled_project(fetched)
    out = fetched.root / "oof.csv"
    assert fetched.run(
        "cv", "--model", "ranker", "--pseudo", "unlabeled", "--pseudo", "train", "--out", str(out)
    ) == 0
    oof = pd.read_csv(out, dtype={"client_id": str})
    assert list(oof.columns) == ["client_id", "fold", *LABELS]
    assert sorted(oof["client_id"]) == sorted(train)  # once each, and no unlabelled Client
    assert np.allclose(oof[LABELS].sum(axis=1), 1.0)
    printed = capsys.readouterr().out
    assert f"for {len(train)} real-labelled Clients" in printed
    assert "unlabeled@2025-10-03" in printed


def test_the_decision_layer_is_fitted_on_real_labelled_out_of_fold_rows_only(fetched, capsys):
    train, _ = pooled_project(fetched)
    assert fetched.run(
        "train", "--model", "ranker", "--decision", "tuned", "--pseudo", "unlabeled", "--pseudo", "train"
    ) == 0
    assert "decision layer tuned on 5-fold out-of-fold probabilities of the real-labelled Clients" in (
        capsys.readouterr().out
    )
    decision = json.loads((fetched.root / "artifacts" / "ranker.decision.json").read_text())
    assert sorted(decision["fitted_on_clients"]) == sorted(train)

    assert fetched.run("evaluate", "--model", "ranker", "--decision", "tuned") == 0
    assert read_rows(fetched.log)[-1]["model"] == "ranker+pseudo+tuned"


def test_pseudo_labels_from_valid_and_test_read_no_label_file(fetched):
    pooled_project(fetched)
    (fetched.raw / "valid_labels.csv").unlink()
    assert fetched.run("train", "--model", "ranker", "--pseudo", "valid", "--pseudo", "test") == 0
    assert [s["split"] for s in meta(fetched)["pseudo"]["sources"]] == ["valid", "test"]


# --- the log row ----------------------------------------------------------------------------


def test_the_log_row_records_the_sources_shifted_cutoffs_weight_and_training_clients(fetched):
    train, unlabeled = pooled_project(fetched)
    assert fetched.run(
        "train", "--model", "ranker", "--pseudo", "unlabeled", "--pseudo", "train:2025-09-01",
        "--pseudo-weight", "0.25",
    ) == 0
    assert fetched.run("evaluate", "--model", "ranker", "--change", "pooled") == 0
    row = read_rows(fetched.log)[-1]
    assert row["pseudo_sources"] == "unlabeled@2025-10-03;train@2025-09-01"
    assert row["pseudo_weight"] == "0.25"
    assert row["pseudo_min_payments"] == "4"
    assert row["training_clients"] == str(2 * len(train) + len(unlabeled))


def test_a_real_label_run_leaves_the_pseudo_label_columns_blank(fetched):
    train, _ = pooled_project(fetched)
    assert fetched.run("train", "--model", "ranker") == 0
    assert fetched.run("evaluate", "--model", "ranker") == 0
    row = read_rows(fetched.log)[-1]
    assert row["model"] == "ranker"
    assert (row["pseudo_sources"], row["pseudo_weight"], row["fidelity"]) == ("", "", "")
    assert row["training_clients"] == str(len(train))


def test_a_failed_fidelity_check_does_not_stop_training_but_the_log_row_says_so(fetched, capsys):
    pooled_project(fetched)
    add_fidelity_row(fetched, "pass", "20260101T000000-aaaaaa")
    add_fidelity_row(fetched, "fail", "20260102T000000-bbbbbb")  # the latest
    assert fetched.run("train", "--model", "ranker", "--pseudo", "unlabeled") == 0
    assert "fidelity check 20260102T000000-bbbbbb failed" in capsys.readouterr().out
    assert meta(fetched)["pseudo"]["fidelity"] == {"run_id": "20260102T000000-bbbbbb", "verdict": "fail"}

    assert fetched.run("evaluate", "--model", "ranker") == 0
    row = read_rows(fetched.log)[-1]
    assert row["fidelity"] == "fail 20260102T000000-bbbbbb"
    assert "fidelity check 20260102T000000-bbbbbb failed" in capsys.readouterr().out


def test_a_passed_fidelity_check_is_noted_on_the_log_row(fetched):
    pooled_project(fetched)
    add_fidelity_row(fetched, "fail", "20260101T000000-aaaaaa")
    add_fidelity_row(fetched, "pass", "20260102T000000-bbbbbb")
    assert fetched.run("train", "--model", "ranker", "--pseudo", "unlabeled") == 0
    assert fetched.run("evaluate", "--model", "ranker") == 0
    assert read_rows(fetched.log)[-1]["fidelity"] == "pass 20260102T000000-bbbbbb"


def test_training_with_pseudo_labels_before_any_fidelity_check_says_it_is_unchecked(fetched, capsys):
    pooled_project(fetched)
    assert fetched.run("train", "--model", "ranker", "--pseudo", "unlabeled") == 0
    assert "no fidelity check logged" in capsys.readouterr().out
    assert fetched.run("evaluate", "--model", "ranker") == 0
    assert read_rows(fetched.log)[-1]["fidelity"] == "unchecked"


# --- refusals -------------------------------------------------------------------------------


def test_a_shifted_cutoff_whose_horizon_is_not_fully_observed_is_refused(fetched, capsys):
    pooled_project(fetched)
    assert fetched.run("train", "--model", "ranker", "--pseudo", "unlabeled:2025-10-04") != 0
    assert "latest Shifted Cutoff is 2025-10-03" in capsys.readouterr().err
    assert not (fetched.root / "artifacts" / "ranker.json").exists()


@pytest.mark.parametrize(
    "args",
    [
        ("train", "--model", "lgbm", "--pseudo", "unlabeled"),
        ("cv", "--model", "rules", "--pseudo", "unlabeled"),
        ("train", "--model", "ranker", "--pseudo-weight", "0.3"),
        ("train", "--model", "ranker", "--pseudo", "unlabeled", "--pseudo-weight", "0"),
        ("train", "--model", "ranker", "--pseudo", "unlabeled", "--pseudo-weight", "-1"),
        ("train", "--model", "ranker", "--pseudo", "labels"),
        ("train", "--model", "ranker", "--pseudo", "unlabeled:yesterday"),
        ("train", "--model", "ranker", "--pseudo", "unlabeled", "--pseudo-min-payments", "1"),
    ],
)
def test_bad_pseudo_label_options_are_refused(fetched, args):
    with pytest.raises(SystemExit) as e:
        fetched.run(*args)
    assert e.value.code != 0
