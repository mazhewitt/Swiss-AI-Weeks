"""The blend of the gated rule and the Stream Ranker (`--model blend`) through the CLI against fixture
data (Seam 1): its probabilities are `rule_weight * rule + (1 - rule_weight) * ranker`, each part takes
its own settings from `train` and `cv`, and the saved blend reloads to the same probabilities."""

import numpy as np
import pandas as pd
import pytest

from recurring_family.blend import BlendModel
from test_lgbm_pipeline import LABELS, read_rows, separable_split
from test_ranker import read_proba
from test_ranker_pseudo_labels import pseudo_histories
from test_rules_baseline import GATE, GATE_LABELS, all_transactions, write_train
from test_lgbm_pipeline import shop, write_labels, write_transactions


def evaluate_proba(project, model, name, *extra):
    out = project.root / f"{name}.csv"
    assert project.run("evaluate", "--model", model, "--proba", str(out), "--change", name, *extra) == 0
    return read_proba(out)


def train_predictions(project, model, *train_args):
    """Train `model`, score it on the train split; return (log row, per-Client predictions)."""
    assert project.run("train", "--model", model, *train_args) == 0
    assert project.run("evaluate", "--model", model, "--split", "train", "--change", model) == 0
    row = read_rows(project.log)[-1]
    saved = pd.read_csv(project.log.parent / "runs" / f"{row['run_id']}.csv", dtype=str, keep_default_na=False)
    return row, dict(zip(saved["client_id"], saved["predicted"]))


@pytest.fixture
def separable(fetched):
    rng = np.random.default_rng(3)
    separable_split(fetched, "train", "T", 12, rng)
    separable_split(fetched, "valid", "V", 8, rng)
    return fetched


@pytest.fixture
def gate_train(fetched):
    write_train(fetched, all_transactions(GATE), GATE_LABELS)
    return fetched


# --- the blended probabilities ------------------------------------------------------------


def test_the_blend_is_the_weighted_average_of_the_rule_and_the_ranker(separable):
    assert separable.run("train", "--model", "rules", "--none-gate") == 0
    rule = evaluate_proba(separable, "rules", "rule")
    assert separable.run("train", "--model", "ranker") == 0
    ranker = evaluate_proba(separable, "ranker", "ranker")
    assert separable.run("train", "--model", "blend", "--none-gate", "--rule-weight", "0.3") == 0
    blend = evaluate_proba(separable, "blend", "blend")

    assert list(blend.columns) == LABELS
    assert set(blend.index) == set(rule.index) == set(ranker.index)
    expected = 0.3 * rule.loc[blend.index] + 0.7 * ranker.loc[blend.index]
    np.testing.assert_allclose(blend.to_numpy(), expected.to_numpy(), atol=1e-9)
    np.testing.assert_allclose(blend.sum(axis=1), 1.0)
    # the parts disagree somewhere, so the check above is not comparing two copies of one model
    assert not np.allclose(rule.to_numpy(), ranker.loc[rule.index].to_numpy())


def test_rule_weight_zero_is_the_ranker(separable):
    assert separable.run("train", "--model", "ranker") == 0
    ranker = evaluate_proba(separable, "ranker", "ranker")
    assert separable.run("train", "--model", "blend", "--rule-weight", "0") == 0
    blend = evaluate_proba(separable, "blend", "blend")
    np.testing.assert_allclose(blend.to_numpy(), ranker.loc[blend.index].to_numpy(), atol=1e-9)


def test_the_gate_reaches_the_blends_rule_and_rule_weight_one_is_the_gated_rule(gate_train):
    _, rule = train_predictions(gate_train, "rules", "--none-gate")
    row, blend = train_predictions(gate_train, "blend", "--none-gate", "--rule-weight", "1")
    assert blend == rule
    assert (blend["G_SHORT"], blend["G_FOUR"], blend["G_FIVE"]) == ("none", "none", "cloud")
    assert row["model"] == "blend+gate4+w1"
    _, ungated = train_predictions(gate_train, "blend", "--rule-weight", "1")
    assert (ungated["G_SHORT"], ungated["G_FOUR"]) == ("gym", "cloud")


def test_a_client_without_streams_gets_a_none_dominated_row(fetched):
    rng = np.random.default_rng(3)
    separable_split(fetched, "train", "T", 12, rng)
    write_transactions(fetched, "valid", [row for c in ("V1", "V2") for row in shop(c)])
    write_labels(fetched, "valid", {"V1": "none", "V2": "none"})
    assert fetched.run("split") == 0
    assert fetched.run("train", "--model", "blend", "--none-gate") == 0
    proba = evaluate_proba(fetched, "blend", "blend")
    assert len(proba) >= 1
    assert (proba.idxmax(axis=1) == "none").all(), proba


# --- saving, settings and the log --------------------------------------------------------


def test_the_saved_blend_keeps_its_rule_weight_and_gate_and_reloads_to_the_same_probabilities(separable):
    assert separable.run("train", "--model", "blend", "--none-gate", "3", "--rule-weight", "0.25") == 0
    first = evaluate_proba(separable, "blend", "first")
    second = evaluate_proba(separable, "blend", "second")
    pd.testing.assert_frame_equal(first, second)

    model = BlendModel.load(separable.root / "artifacts" / "blend.json")
    assert model.rule_weight == 0.25 and model.rules.none_gate == 3
    assert read_rows(separable.log)[-1]["model"] == "blend+gate3+w0.25"


def test_the_default_rule_weight_is_one_half(separable):
    assert separable.run("train", "--model", "blend") == 0
    assert BlendModel.load(separable.root / "artifacts" / "blend.json").rule_weight == 0.5


def test_cv_builds_the_blend_with_its_settings(gate_train):
    rules_out, blend_out = gate_train.root / "rules_oof.csv", gate_train.root / "blend_oof.csv"
    plain_out = gate_train.root / "plain_oof.csv"
    args = ("--folds", "2")
    assert gate_train.run("cv", "--model", "rules", "--none-gate", "--out", str(rules_out), *args) == 0
    assert gate_train.run("cv", "--model", "rules", "--out", str(plain_out), *args) == 0
    assert gate_train.run(
        "cv", "--model", "blend", "--none-gate", "--rule-weight", "1", "--out", str(blend_out), *args
    ) == 0
    rules, blend, plain = (read_proba(p) for p in (rules_out, blend_out, plain_out))
    pd.testing.assert_frame_equal(blend, rules)
    assert rules.loc["G_SHORT", "none"] == 1.0 and plain.loc["G_SHORT", "gym"] == 1.0


def test_the_tuned_blend_writes_a_valid_submission(separable, capsys):
    assert separable.run("train", "--model", "blend", "--none-gate", "--decision", "tuned", "--folds", "2") == 0
    assert "decision layer tuned" in capsys.readouterr().out
    assert separable.run("submit", "--model", "blend", "--decision", "tuned", "--name", "blend") == 0
    assert separable.run("submit", "--check", str(separable.root / "submissions" / "blend.csv")) == 0


def test_pseudo_labels_reach_the_blends_ranker(fetched):
    # every real train label is none: only the unlabelled Clients' Pseudo-Labels can teach the ranker
    write_transactions(fetched, "train", [row for c in ("N1", "N2", "N3") for row in shop(c)])
    write_labels(fetched, "train", {"N1": "none", "N2": "none", "N3": "none"})
    separable_split(fetched, "valid", "V", 8, np.random.default_rng(4))
    pseudo_histories(fetched, "unlabeled", "U")

    assert fetched.run("train", "--model", "blend", "--rule-weight", "0", "--pseudo", "unlabeled") == 0
    assert fetched.run("evaluate", "--model", "blend", "--change", "blend with Pseudo-Labels") == 0
    row = read_rows(fetched.log)[-1]
    assert float(row["macro_f1"]) >= 0.9, row
    assert row["model"] == "blend+w0+pseudo"
    assert row["pseudo_sources"] == "unlabeled@2025-10-03"


# --- refusals ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["train", "--model", "ranker", "--rule-weight", "0.5"],
        ["train", "--model", "rules", "--rule-weight", "0.5"],
        ["train", "--model", "blend", "--rule-weight", "1.5"],
        ["train", "--model", "blend", "--rule-weight", "-0.1"],
        ["train", "--model", "blend", "--rule-weight", "half"],
        ["train", "--model", "ranker", "--none-gate"],
        ["cv", "--model", "prior", "--none-gate"],
        ["train", "--model", "rules", "--pseudo", "unlabeled"],
    ],
)
def test_options_for_other_models_and_bad_rule_weights_are_refused(fetched, args):
    with pytest.raises(SystemExit):
        fetched.run(*args)
    assert not list(fetched.root.glob("artifacts/*.json"))
