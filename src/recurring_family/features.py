"""E2 features: one row per Client, built only from its Recurring Streams (ADR 0001).

A row has three parts, all from the stream table, never from raw transactions:

- the top three Active Stream slots, soonest projected next payment first: family, MCC, amount,
  period, regularity (gap spread and amount variation), days to next payment, payment count and
  the description rate below;
- a block per Merchant Family over its Active Streams and its streams with 3+ payments;
- the `none` signals: stream counts, the longest Active Stream, the earliest Active Stream start and
  the share of family-specific descriptions among Active Stream payments.

Streams with fewer than three payments never enter a feature: their number shifts between splits
because description noise on non-home MCCs creates short fragments at different rates.

The one label-derived feature is the description rate: how often an Active Stream with a given
description belongs to its Client's Next Recurring Family, smoothed towards the overall rate.
Descriptions never seen in training (test-only noise) encode as unknown, which is that overall rate.
On training Clients it is fitted out of fold, so no Client's own label reaches its features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import CUTOFF, MERCHANT_FAMILIES
from .streams import HOME_MCC

SLOTS = 3
ESTABLISHED_PAYMENTS = 3  # streams with fewer payments never enter a feature
UNKNOWN = "unknown"
KNOWN_MCCS = tuple(sorted(HOME_MCC))

SLOT_FIELDS = (
    "family", "mcc", "amount", "period_days", "gap_mad_days", "amount_cv", "days_to_next", "n_payments",
    "description_rate",
)
FAMILY_FIELDS = ("active_streams", "max_payments", "days_to_next", "days_since_last", "days_since_first")
NONE_SIGNALS = (
    "n_active_streams", "n_streams_3plus", "longest_active_payments", "earliest_active_start_days",
    "family_description_share",
)

FEATURE_COLUMNS = [
    *(f"slot{k}_{field}" for k in range(1, SLOTS + 1) for field in SLOT_FIELDS),
    *(f"{family}_{field}" for family in MERCHANT_FAMILIES for field in FAMILY_FIELDS),
    *NONE_SIGNALS,
]
# Fixed category lists, so every split (and every fold) gets the same encoding.
CATEGORIES = {
    **{f"slot{k}_family": list(MERCHANT_FAMILIES) for k in range(1, SLOTS + 1)},
    **{f"slot{k}_mcc": [*KNOWN_MCCS, UNKNOWN] for k in range(1, SLOTS + 1)},
}
_COUNTS = [f"{family}_{field}" for family in MERCHANT_FAMILIES for field in ("active_streams", "max_payments")]
_COUNTS += ["n_active_streams", "n_streams_3plus", "longest_active_payments"]

DESCRIPTION_SMOOTHING = 10.0  # pseudo-streams at the overall rate added to every description
DESCRIPTION_FOLDS = 5
DESCRIPTION_SEED = 0

_DAY = pd.Timedelta(days=1)


def client_features(
    streams: pd.DataFrame, clients: pd.Index, description_rate: pd.Series, cutoff: pd.Timestamp = CUTOFF
) -> pd.DataFrame:
    """One row per Client in `clients` (in that order) with exactly `FEATURE_COLUMNS`.

    `description_rate` is aligned with `streams`' index. Clients without streams get empty slots,
    zero counts and missing values elsewhere.
    """
    clients = pd.Index(clients, name="client_id")
    s = streams[streams["client_id"].isin(set(clients))].copy()
    s["client_id"] = s["client_id"].astype(str)
    s["description_rate"] = description_rate.reindex(s.index).astype(float)
    s["amount"] = s["median_amount"]
    s["days_to_next"] = (s["next_payment"] - cutoff) / _DAY
    s["days_since_last"] = (cutoff - s["last_payment"]) / _DAY
    s["days_since_first"] = (cutoff - s["first_payment"]) / _DAY
    s["mcc"] = s["mcc"].astype(object).where(s["mcc"].isin(KNOWN_MCCS), UNKNOWN)
    active = s[s["active"]]
    established = s[s["n_payments"] >= ESTABLISHED_PAYMENTS]

    parts = [_slots(active), _family_block(active, established), _none_signals(active, established)]
    out = pd.concat(parts, axis=1).reindex(index=clients, columns=FEATURE_COLUMNS)
    out[_COUNTS] = out[_COUNTS].fillna(0)
    for column in FEATURE_COLUMNS:
        if column in CATEGORIES:
            out[column] = pd.Categorical(out[column], categories=CATEGORIES[column])
        else:
            out[column] = out[column].astype(float)
    return out


def _slots(active: pd.DataFrame) -> pd.DataFrame:
    if active.empty:
        return pd.DataFrame()
    ranked = active.sort_values(
        ["client_id", "days_to_next", "n_payments", "family", "median_amount"],
        ascending=[True, True, False, True, True],
    )
    ranked = ranked.assign(slot=ranked.groupby("client_id").cumcount() + 1)
    ranked = ranked[ranked["slot"] <= SLOTS]
    wide = ranked.set_index(["client_id", "slot"])[list(SLOT_FIELDS)].unstack("slot")
    wide.columns = [f"slot{k}_{field}" for field, k in wide.columns]
    return wide


def _family_block(active: pd.DataFrame, established: pd.DataFrame) -> pd.DataFrame:
    per_active = active.groupby(["client_id", "family"]).agg(
        active_streams=("stream_id", "size"), days_to_next=("days_to_next", "min")
    )
    per_established = established.groupby(["client_id", "family"]).agg(
        max_payments=("n_payments", "max"),
        days_since_last=("days_since_last", "min"),
        days_since_first=("days_since_first", "max"),
    )
    both = per_active.join(per_established, how="outer")
    if both.empty:
        return pd.DataFrame()
    wide = both.unstack("family")
    wide.columns = [f"{family}_{field}" for field, family in wide.columns]
    return wide


def _none_signals(active: pd.DataFrame, established: pd.DataFrame) -> pd.DataFrame:
    specific = active["family_description_share"] * active["n_payments"]
    per_client = active.assign(specific=specific).groupby("client_id").agg(
        n_active_streams=("stream_id", "size"),
        longest_active_payments=("n_payments", "max"),
        earliest_active_start_days=("days_since_first", "max"),
        specific=("specific", "sum"),
        payments=("n_payments", "sum"),
    )
    per_client["family_description_share"] = per_client.pop("specific") / per_client.pop("payments")
    per_client["n_streams_3plus"] = established.groupby("client_id").size()
    return per_client


# --- the label-derived description rate -------------------------------------------


def _stream_outcomes(streams: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Active Streams of labelled Clients with a flag: is the stream's family the Client's label?"""
    active = streams[streams["active"] & streams["client_id"].isin(set(labels.index))]
    hit = active["family"].to_numpy(dtype=object) == labels.reindex(active["client_id"].astype(str)).to_numpy(dtype=object)
    return pd.DataFrame({"description": active["description"].astype(str), "hit": hit.astype(float)}, index=active.index)


class DescriptionRates:
    """Smoothed per-description rate at which an Active Stream is its Client's Next Recurring Family."""

    def __init__(self, rates: dict[str, float], prior: float):
        self.rates = rates
        self.prior = prior

    @classmethod
    def fit(cls, streams: pd.DataFrame, labels: pd.Series, prior: float | None = None) -> "DescriptionRates":
        outcomes = _stream_outcomes(streams, labels)
        if prior is None:
            prior = float(outcomes["hit"].mean()) if len(outcomes) else 0.0
        grouped = outcomes.groupby("description")["hit"].agg(["sum", "count"])
        rates = (grouped["sum"] + DESCRIPTION_SMOOTHING * prior) / (grouped["count"] + DESCRIPTION_SMOOTHING)
        return cls({str(k): float(v) for k, v in rates.items()}, prior)

    def encode(self, streams: pd.DataFrame) -> pd.Series:
        """Each stream's rate; a description never seen in training is unknown: the overall rate."""
        mapped = streams["description"].astype(object).map(self.rates)
        return mapped.fillna(self.prior).astype(float)

    def to_dict(self) -> dict:
        return {"rates": self.rates, "prior": self.prior}

    @classmethod
    def from_dict(cls, d: dict) -> "DescriptionRates":
        return cls({str(k): float(v) for k, v in d["rates"].items()}, float(d["prior"]))


def out_of_fold_description_rates(streams: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Rates for the labelled Clients' streams, each from the other folds' Clients only, so a
    training row never sees its own label. Unseen descriptions get the overall rate."""
    prior = DescriptionRates.fit(streams, labels).prior
    rates = pd.Series(prior, index=streams.index, dtype=float)
    clients = pd.Index(labels.index)
    if len(clients) < 2:
        return rates
    splitter = KFold(n_splits=min(DESCRIPTION_FOLDS, len(clients)), shuffle=True, random_state=DESCRIPTION_SEED)
    client_of = streams["client_id"].astype(str)
    for fit_idx, encode_idx in splitter.split(np.arange(len(clients))):
        fitted = DescriptionRates.fit(streams, labels.iloc[fit_idx], prior=prior)
        scored = client_of.isin(set(clients[encode_idx]))
        rates[scored] = fitted.encode(streams[scored]).to_numpy()
    return rates
