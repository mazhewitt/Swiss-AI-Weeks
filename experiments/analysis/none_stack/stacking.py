"""Ticket 19's pure helpers (no data, no labels): within-domain percentile ranks, the stacker's input, and the
probability adjustment that puts the stacker's P'(none) into B's eight-label probabilities."""
import numpy as np
import pandas as pd

from recurring_family.config import LABELS, MERCHANT_FAMILIES, NONE_LABEL

LOGIT_COLUMN = "logit_pb_none"
CLIP = 1e-6
# P_B(none) at or above 1 - this is treated as P_B(none) = 1 (no family mass to rescale)
NONE_ONE_TOLERANCE = 1e-12


def rank_within_domain(raw: pd.DataFrame) -> pd.DataFrame:
    """Each column as its percentile rank among this domain's Clients (the rows given): average ranks for
    ties, divided by the number of non-missing values, so in (0, 1]. Missing values stay missing."""
    return raw.rank(method="average", pct=True, na_option="keep").astype(float)


def logit(p: pd.Series) -> pd.Series:
    q = p.astype(float).clip(CLIP, 1 - CLIP)
    return np.log(q / (1 - q))


def stacker_input(base: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """[logit P_B(none)] plus the raw block ranked within this domain, for the Clients of `raw`."""
    ranked = rank_within_domain(raw)
    lp = logit(base[NONE_LABEL].reindex(ranked.index)).rename(LOGIT_COLUMN)
    assert lp.notna().all(), "B has no probabilities for some Clients"
    return pd.concat([lp, ranked], axis=1)


def adjust(base: pd.DataFrame, p_none: pd.Series) -> pd.DataFrame:
    """P'(none) = the stacker's output; each family's P_B(family) scaled by (1 - P'(none)) / (1 - P_B(none)).
    Where P_B(none) = 1 there is no family mass to scale, so the (1 - P'(none)) is split equally."""
    base = base[list(LABELS)].astype(float)
    p_new = p_none.reindex(base.index).astype(float)
    assert p_new.notna().all(), "no P'(none) for some Clients"
    rest = 1.0 - base[NONE_LABEL]
    degenerate = rest <= NONE_ONE_TOLERANCE
    scale = (1.0 - p_new) / rest.where(~degenerate, 1.0)
    families = base[list(MERCHANT_FAMILIES)].mul(scale, axis=0)
    equal = (1.0 - p_new) / len(MERCHANT_FAMILIES)
    families = families.where(~degenerate, pd.DataFrame({f: equal for f in MERCHANT_FAMILIES}))
    out = families.assign(**{NONE_LABEL: p_new})
    return out[list(LABELS)]
