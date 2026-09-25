"""Ticket 19's pure helpers (`experiments/analysis/none_stack/stacking.py`): the probability adjustment that
puts the stacker's P'(none) into B's probabilities, and the within-domain percentile ranks."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recurring_family.config import LABELS, MERCHANT_FAMILIES, NONE_LABEL

_PATH = Path(__file__).resolve().parent.parent / "experiments" / "analysis" / "none_stack" / "stacking.py"
_spec = importlib.util.spec_from_file_location("none_stack_stacking", _PATH)
stacking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stacking)


def proba(rows: dict[str, list[float]]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index", columns=list(LABELS))
    df.index.name = "client_id"
    return df


BASE = proba({
    "a": [0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.6],
    "b": [0.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.05, 0.2],
    "c": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # P_B(none) = 1: no family mass
    "d": [0.3, 0.0, 0.0, 0.4, 0.0, 0.3, 0.0, 0.0],
})
P_NEW = pd.Series({"a": 0.2, "b": 0.6, "c": 0.3, "d": 0.5})


def test_families_rescale_by_the_ratio_of_family_mass():
    out = stacking.adjust(BASE, P_NEW)
    for c in ("a", "b", "d"):
        ratio = (1 - P_NEW[c]) / (1 - BASE.loc[c, NONE_LABEL])
        np.testing.assert_allclose(out.loc[c, list(MERCHANT_FAMILIES)], BASE.loc[c, list(MERCHANT_FAMILIES)] * ratio)
    # the families keep their relative order and zeros stay zero
    assert out.loc["a", "cloud"] == pytest.approx(2 * out.loc["a", "gym"])
    assert out.loc["b", "cloud"] == 0.0


def test_none_is_the_stackers_output_and_rows_sum_to_one():
    out = stacking.adjust(BASE, P_NEW)
    assert list(out.columns) == list(LABELS)
    pd.testing.assert_series_equal(out[NONE_LABEL], P_NEW.reindex(out.index), check_names=False)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert (out.to_numpy() >= 0).all()


def test_p_b_none_of_one_splits_the_family_mass_equally():
    out = stacking.adjust(BASE, P_NEW)
    row = out.loc["c"]
    np.testing.assert_allclose(row[list(MERCHANT_FAMILIES)], (1 - 0.3) / len(MERCHANT_FAMILIES))
    assert row[NONE_LABEL] == pytest.approx(0.3)
    assert np.isfinite(out.to_numpy()).all()


def test_unchanged_p_none_leaves_b_as_it_is():
    out = stacking.adjust(BASE.drop(index="c"), BASE[NONE_LABEL].drop("c"))
    np.testing.assert_allclose(out.to_numpy(), BASE.drop(index="c").to_numpy())


def test_ranks_are_within_the_domain_given():
    selection = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4]}, index=list("abcd"))
    test = selection + 10.0  # a level shift, such as the Decoy share's drift
    np.testing.assert_allclose(stacking.rank_within_domain(selection)["x"], [0.25, 0.5, 0.75, 1.0])
    pd.testing.assert_frame_equal(stacking.rank_within_domain(test), stacking.rank_within_domain(selection))


def test_ranks_average_ties_and_keep_missing_values_missing():
    raw = pd.DataFrame({"x": [1.0, 1.0, np.nan, 3.0], "y": [5.0, 4.0, 3.0, 2.0]}, index=list("abcd"))
    r = stacking.rank_within_domain(raw)
    np.testing.assert_allclose(r.loc[["a", "b", "d"], "x"], [0.5, 0.5, 1.0])  # 3 non-missing values
    assert np.isnan(r.loc["c", "x"])
    np.testing.assert_allclose(r["y"], [1.0, 0.75, 0.5, 0.25])


def test_stacker_input_is_the_logit_plus_the_ranked_block():
    raw = pd.DataFrame({"x": [3.0, 1.0, 2.0, 4.0]}, index=list("abcd"))
    x = stacking.stacker_input(BASE, raw)
    assert list(x.columns) == [stacking.LOGIT_COLUMN, "x"]
    assert x.loc["a", stacking.LOGIT_COLUMN] == pytest.approx(np.log(0.6 / 0.4))
    assert np.isfinite(x[stacking.LOGIT_COLUMN]).all()  # P_B(none) of 0 and 1 are clipped
    np.testing.assert_allclose(x["x"], [0.75, 0.25, 0.5, 1.0])  # the raw block arrives ranked, not raw
