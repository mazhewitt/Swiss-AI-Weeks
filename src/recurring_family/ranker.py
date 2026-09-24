"""The Stream Ranker: which of a Client's Recurring Streams carries its Next Recurring Family?

Every detected Recurring Stream is a Candidate Stream, whether or not it is Active. A LightGBM
binary model scores each candidate: how likely is its family the Client's label? The features
come only from the stream table (ADR 0001): the candidate's own fields, its rank among the
Client's streams by projected next payment, and Client-level context. No description statistic
and no label-derived encoding enters a feature.

Per Client, each Merchant Family's score is its best candidate's score, `none` is one minus the
best score overall, and the row is normalised to sum to 1. A Client without candidates is `none`.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CUTOFF, LABELS, MERCHANT_FAMILIES, NONE_LABEL
from .streams import detect_streams

CANDIDATE_FIELDS = (
    "family",
    "days_to_next",
    "next_rank",
    "n_payments",
    "days_since_first",
    "days_since_last",
    "period_days",
    "gap_mad_days",
    "amount_cv",
    "amount",
    "refund_rate",
    "active",
)
CLIENT_CONTEXT = ("n_streams", "n_active_streams", "longest_active_payments", "earliest_active_start_days")
FEATURE_COLUMNS = [*CANDIDATE_FIELDS, *CLIENT_CONTEXT]

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": 0,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}

_DAY = pd.Timedelta(days=1)


# --- Candidate Streams ------------------------------------------------------------------


def candidates(streams: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF) -> pd.DataFrame:
    """One row per Candidate Stream: `client_id` plus exactly `FEATURE_COLUMNS`, from the stream table only."""
    s = streams.reset_index(drop=True)
    out = pd.DataFrame({"client_id": s["client_id"].astype(str).to_numpy()})
    out["family"] = pd.Categorical(s["family"].astype(str).to_numpy(), categories=list(MERCHANT_FAMILIES))
    out["days_to_next"] = ((s["next_payment"] - cutoff) / _DAY).to_numpy(dtype=float)
    # 1 = the Client's soonest projected next payment; streams with no projection (one payment) come last
    ranked = out.assign(_order=out["days_to_next"].fillna(np.inf), _n=-s["n_payments"].to_numpy())
    out["next_rank"] = (
        ranked.sort_values(["client_id", "_order", "_n"], kind="stable").groupby("client_id").cumcount() + 1
    ).reindex(out.index).astype(float)
    out["n_payments"] = s["n_payments"].to_numpy(dtype=float)
    out["days_since_first"] = ((cutoff - s["first_payment"]) / _DAY).to_numpy(dtype=float)
    out["days_since_last"] = ((cutoff - s["last_payment"]) / _DAY).to_numpy(dtype=float)
    out["period_days"] = s["period_days"].to_numpy(dtype=float)
    out["gap_mad_days"] = s["gap_mad_days"].to_numpy(dtype=float)
    out["amount_cv"] = s["amount_cv"].to_numpy(dtype=float)
    out["amount"] = s["median_amount"].to_numpy(dtype=float)
    out["refund_rate"] = s["refund_rate"].to_numpy(dtype=float)
    out["active"] = s["active"].to_numpy(dtype=float)

    active = out[out["active"] == 1.0]
    per_client = pd.DataFrame(
        {
            "n_streams": out.groupby("client_id").size(),
            "n_active_streams": active.groupby("client_id").size(),
            "longest_active_payments": active.groupby("client_id")["n_payments"].max(),
            "earliest_active_start_days": active.groupby("client_id")["days_since_first"].max(),
        }
    )
    context = per_client.reindex(out["client_id"]).reset_index(drop=True)
    context[["n_active_streams", "longest_active_payments"]] = context[
        ["n_active_streams", "longest_active_payments"]
    ].fillna(0)
    for column in CLIENT_CONTEXT:
        out[column] = context[column].to_numpy(dtype=float)
    return out[["client_id", *FEATURE_COLUMNS]]


def client_proba(scored: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
    """Candidate scores (columns client_id, family, score) -> one row per Client in `clients` with
    exactly `LABELS`: each family its best candidate's score, `none` one minus the best score, the
    row normalised to sum to 1. A Client without candidates is certainly `none`."""
    clients = pd.Index(clients, name="client_id")
    best = (
        scored.assign(client_id=scored["client_id"].astype(str), family=scored["family"].astype(str))
        .groupby(["client_id", "family"])["score"]
        .max()
        .unstack("family")
    )
    families = best.reindex(index=clients.astype(str), columns=list(MERCHANT_FAMILIES)).fillna(0.0)
    families.index = clients
    out = families.astype(float)
    out[NONE_LABEL] = 1.0 - out.max(axis=1)
    out = out[list(LABELS)]
    return out.div(out.sum(axis=1), axis=0)


# --- stream detection, once per Client history --------------------------------------------


class _StreamMemo:
    """Detected streams per Client history, so cross-validation detects each Client once.

    A Client's history is keyed by its transactions' content, so any change to them (an injected
    decoy, a truncated history) is detected afresh. Only streams are kept, never labels.
    """

    def __init__(self):
        self.table: pd.DataFrame | None = None  # the stream table plus each stream's history key
        self.seen: set[str] = set()

    def streams(self, transactions: pd.DataFrame) -> pd.DataFrame:
        keys = _history_keys(transactions)
        missing = keys[~keys.isin(self.seen)]
        if len(missing):
            fresh = detect_streams(transactions[transactions["client_id"].astype(str).isin(set(missing.index))])
            fresh["_key"] = fresh["client_id"].astype(str).map(missing).to_numpy()
            if self.table is None:
                self.table = fresh
            elif len(fresh):
                self.table = pd.concat([self.table, fresh], ignore_index=True)
            self.seen.update(missing)
        if self.table is None:  # no Client history seen yet: an empty, typed stream table
            return detect_streams(transactions)
        found = self.table[self.table["_key"].isin(set(keys))]
        return found.drop(columns="_key").sort_values(["client_id", "stream_id"]).reset_index(drop=True)


_PRIMES = (1_000_000_007, 998_244_353)


def _history_keys(transactions: pd.DataFrame) -> pd.Series:
    """One content key per Client (client_id -> key), independent of row order."""
    client = transactions["client_id"].astype(str)
    if transactions.empty:
        return pd.Series(dtype=object)
    hashed = pd.util.hash_pandas_object(transactions[sorted(transactions.columns)], index=False).to_numpy()
    parts = pd.DataFrame({"client_id": client.to_numpy()})
    for i, p in enumerate(_PRIMES):
        parts[f"h{i}"] = (hashed % np.uint64(p)).astype(np.int64)
    summed = parts.groupby("client_id").agg(n=("h0", "size"), h0=("h0", "sum"), h1=("h1", "sum"))
    return pd.Series(
        [f"{c}:{n}:{a}:{b}" for c, n, a, b in zip(summed.index, summed["n"], summed["h0"], summed["h1"])],
        index=summed.index,
        dtype=object,
    )


_MEMO = _StreamMemo()


def _streams_of(transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
    return _MEMO.streams(transactions[transactions["client_id"].astype(str).isin(set(map(str, clients)))])


# --- the model ------------------------------------------------------------------------------


class RankerModel:
    """Scores every Candidate Stream with a LightGBM binary model; see the module docstring."""

    name = "ranker"

    def __init__(self, booster: lgb.Booster | None = None, constant: float | None = None):
        self.booster = booster
        self.constant = constant  # the score of every candidate when training had one outcome only

    def fit(self, transactions: pd.DataFrame, labels: pd.Series) -> "RankerModel":
        unknown = set(labels) - set(LABELS)
        if unknown:
            raise ValueError(f"labels outside the allowed set: {sorted(unknown)}")
        rows = candidates(_streams_of(transactions, labels.index))
        target = (
            rows["family"].astype(str).to_numpy(dtype=object)
            == labels.astype(str).reindex(rows["client_id"]).to_numpy(dtype=object)
        ).astype(int)
        self.booster, self.constant = None, None
        if len(set(target)) < 2:
            self.constant = float(target.mean()) if len(target) else 0.0
            return self
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(rows[FEATURE_COLUMNS], target)
        self.booster = model.booster_
        return self

    def predict_proba(self, transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
        if self.booster is None and self.constant is None:
            raise RuntimeError("RankerModel is not fitted")
        rows = candidates(_streams_of(transactions, clients))
        if self.booster is None:
            score = np.full(len(rows), self.constant, dtype=float)
        else:
            score = np.asarray(self.booster.predict(rows[FEATURE_COLUMNS]), dtype=float) if len(rows) else np.zeros(0)
        scored = pd.DataFrame({"client_id": rows["client_id"], "family": rows["family"].astype(str), "score": score})
        return client_proba(scored, clients)

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "model": self.name,
                    "features": FEATURE_COLUMNS,
                    "constant": self.constant,
                    "booster": None if self.booster is None else self.booster.model_to_string(),
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "RankerModel":
        saved = json.loads(Path(path).read_text())
        if saved.get("features") != FEATURE_COLUMNS:
            raise ValueError(f"{path} was saved with other ranker features; retrain it")
        booster = None if saved["booster"] is None else lgb.Booster(model_str=saved["booster"])
        return cls(booster, saved["constant"])
