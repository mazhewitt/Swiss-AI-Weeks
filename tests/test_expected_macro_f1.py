"""Ticket 17: the batch expected-macro-F1 decision D, on toy data."""

import numpy as np
import pandas as pd
import pytest

from recurring_family import decision as decision_layer
from recurring_family.evaluation import score

LABELS = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none"]


def proba(rows):
    """Rows of {label: p}; missing labels are 0."""
    df = pd.DataFrame(rows, columns=LABELS).fillna(0.0)
    df.index = pd.Index([f"C{i:03d}" for i in range(len(df))], name="client_id")
    return df


def test_hand_computed_expected_macro_f1():
    p = proba([{"cloud": 0.6, "none": 0.4}, {"cloud": 1.0}])
    predicted = pd.Series("cloud", index=p.index)
    # cloud: E[TP] = 1.6, predicted 2, E[true] = 1.6; none: E[TP] = 0, predicted 0, E[true] = 0.4;
    # the six other labels have nothing predicted and nothing expected, so they score 0
    expected = (2 * 1.6 / (2 + 1.6)) / 8
    assert decision_layer.expected_macro_f1(predicted, p) == pytest.approx(expected)


def test_one_hot_probabilities_give_the_realised_macro_f1():
    rng = np.random.default_rng(0)
    truth = pd.Series(rng.choice(LABELS, size=60), index=[f"C{i:03d}" for i in range(60)])
    p = pd.DataFrame(
        (truth.to_numpy()[:, None] == np.array(LABELS)[None, :]).astype(float), index=truth.index, columns=LABELS
    )
    predicted = pd.Series(rng.choice(LABELS, size=60), index=truth.index)
    assert decision_layer.expected_macro_f1(predicted, p) == pytest.approx(score(truth, predicted)["macro_f1"])


def test_predictions_are_aligned_by_client_not_position():
    p = proba([{"cloud": 1.0}, {"gym": 1.0}])
    predicted = pd.Series({"C001": "gym", "C000": "cloud"})
    assert decision_layer.expected_macro_f1(predicted, p) == pytest.approx(2 / 8)


def test_column_order_does_not_matter():
    p = proba([{"cloud": 0.7, "gym": 0.3}, {"gym": 0.9, "none": 0.1}, {"none": 0.8, "music": 0.2}])
    predicted = pd.Series(["cloud", "gym", "none"], index=p.index)
    shuffled = p[list(reversed(LABELS))]
    assert decision_layer.expected_macro_f1(predicted, shuffled) == pytest.approx(
        decision_layer.expected_macro_f1(predicted, p)
    )


def test_fit_expected_never_scores_below_argmax_and_stays_on_the_grid():
    rng = np.random.default_rng(1)
    raw = rng.dirichlet(np.full(len(LABELS), 0.4), size=200)
    p = pd.DataFrame(raw, columns=LABELS, index=[f"C{i:03d}" for i in range(200)])
    found, info = decision_layer.fit_expected(p)
    argmax = decision_layer.expected_macro_f1(decision_layer.Decision().apply(p), p)
    assert info["argmax_expected_macro_f1"] == pytest.approx(argmax)
    assert info["expected_macro_f1"] >= argmax
    assert info["expected_macro_f1"] == pytest.approx(decision_layer.expected_macro_f1(found.apply(p), p))
    assert set(found.weights.values()) <= set(decision_layer.WEIGHT_GRID)
    assert found.none_threshold is None or found.none_threshold in decision_layer.THRESHOLD_GRID


def test_fit_expected_calls_none_on_a_batch_where_none_is_likely_but_never_the_argmax():
    # every Client has P(none) = 0.45 and its family 0.55: argmax never says none, so none's expected
    # F1 is 0; a none threshold on the most none-like Clients gains expected F1
    rows = [{"cloud": 0.55, "none": 0.45}] * 10 + [{"cloud": 0.9, "none": 0.1}] * 10
    rows += [{"gym": 0.55, "none": 0.45}] * 10 + [{"gym": 0.9, "none": 0.1}] * 10
    p = proba(rows)
    found, info = decision_layer.fit_expected(p)
    assert info["expected_macro_f1"] > info["argmax_expected_macro_f1"]
    assert (found.apply(p) == "none").any()
