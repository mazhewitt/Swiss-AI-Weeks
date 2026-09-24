"""The Stream Ranker's Client-level `none` model (`--none-model`).

The ranker's own `none` is one minus its best candidate's score: it asks whether one stream carries
the label. But at the Cutoff about 23% of train Clients with an Active Stream are `none`, while inside
the history only 3-5% of Active Streams stop within 90 days. So `none` depends on the Client, not on
one stream's timing. This model scores P(`none`) from one row per Client with at least one Candidate
Stream:

- the Client's stream counts, and the candidate fields over its Active Streams: the soonest-due one's
  value, the minimum and the maximum;
- churn signals from its streams' member payments and matched refunds: overdue and missed payments,
  the gap trend, refunds near the end, how many of its streams ended and when, and how its stream
  activity of the last 60 and 90 days compares with before;
- optionally the ranker's own `none` for the Client (`RANKER_NONE`), cross-fitted for training Clients.

Every feature comes from the stream table or the member payments and matched refunds of detected
streams (ADR 0001): never a raw-transaction count and never a description statistic.

`combine` puts it together with the ranker: P(`none`) comes from this model, and each Merchant Family
gets its share of the ranker's family scores times 1 - P(`none`).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CUTOFF, LABELS, MERCHANT_FAMILIES, NONE_LABEL

# the candidate fields aggregated over a Client's Active Streams
CANDIDATE_FIELDS = (
    "n_payments",
    "days_since_first",
    "days_since_last",
    "period_days",
    "gap_mad_days",
    "amount_cv",
    "amount",
    "refund_rate",
    "days_to_next",
)
# per-stream churn signals from member payments and matched refunds, aggregated the same way
CHURN_FIELDS = (
    "overdue",
    "last_gap_ratio",
    "last_gap_missed",
    "last_gap_late",
    "n_missed",
    "missed_rate",
    "missed_last3",
    "gap_trend",
    "gap_recent_minus_all",
    "gap_mad_rel",
    "expected_minus_actual",
    "phase",
    "refund_last",
    "refund_last2",
    "refund_last_days",
    "n_refunds",
    "last_dom",
    "first_dom",
    "next_dom",
)
AGGREGATES = ("soonest", "min", "max")
CLIENT_CHURN = (
    "n_ended",
    "n_short",
    "past_churn_rate",
    "n_ended_recent120",
    "n_families_active",
    "n_families_all",
    "total_payments",
    "stream_first_days",
    "stream_last_days",
    "stream_activity_ratio60",
    "stream_activity_ratio90",
    *(f"active_has_{family}" for family in MERCHANT_FAMILIES),
)


def _aggregated(fields) -> list[str]:
    return [f"{a}_{f}" for f in fields for a in AGGREGATES]


GROUP_A = [*_aggregated(CANDIDATE_FIELDS), "n_streams", "n_active_streams"]
GROUP_B = [*_aggregated(CHURN_FIELDS), *CLIENT_CHURN]
RANKER_NONE = "ranker_none"  # the ranker's own `none` probability, cross-fitted for training Clients
FEATURE_COLUMNS = [*GROUP_A, *GROUP_B]  # every feature `client_features` computes
NONE_FEATURES = [RANKER_NONE, *GROUP_A, *GROUP_B]  # the features the `none` model uses

NONE_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "random_state": 0,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 1,
    "verbose": -1,
}

_DAY = pd.Timedelta(days=1)


# --- features -------------------------------------------------------------------------------


def client_features(streams: pd.DataFrame, payments: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF) -> pd.DataFrame:
    """One row per Client with at least one Recurring Stream (index `client_id`, sorted), with exactly
    `FEATURE_COLUMNS`, from a stream table and its member payments (`detect_stream_payments`) only.
    A Client's row depends on its own streams alone, never on a label."""
    clients = pd.Index(sorted(streams["client_id"].astype(str).unique()), name="client_id")
    if streams.empty:
        return pd.DataFrame(index=clients, columns=FEATURE_COLUMNS, dtype=float)
    s = _stream_fields(streams.sort_values(["client_id", "stream_id"]).reset_index(drop=True), payments, cutoff)
    active = s[s["active"]]
    ended = s[~s["active"] & (s["n_payments"] >= 3)]

    def count(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby("client_id").size().reindex(clients).fillna(0)

    by_client = s.groupby("client_id")
    out = {"n_streams": by_client.size(), "n_active_streams": count(active)}
    # the soonest-due Active Stream (more payments first on a tie), then the minimum and the maximum
    soonest = (
        active.assign(_order=active["days_to_next"].fillna(np.inf), _n=-active["n_payments"])
        .sort_values(["client_id", "_order", "_n", "stream_id"], kind="stable")
        .drop_duplicates("client_id")
        .set_index("client_id")
    )
    lowest, highest = active.groupby("client_id").min(numeric_only=True), active.groupby("client_id").max(numeric_only=True)
    for field in (*CANDIDATE_FIELDS, *CHURN_FIELDS):
        out[f"soonest_{field}"] = soonest[field].reindex(clients)
        out[f"min_{field}"] = lowest[field].reindex(clients)
        out[f"max_{field}"] = highest[field].reindex(clients)

    out["n_ended"] = count(ended)
    out["n_short"] = count(s[s["n_payments"] < 3])
    out["past_churn_rate"] = out["n_ended"] / (out["n_ended"] + out["n_active_streams"]).replace(0, np.nan)
    out["n_ended_recent120"] = count(ended[ended["days_since_last"] < 120])
    out["n_families_active"] = active.groupby("client_id")["family"].nunique().reindex(clients).fillna(0)
    out["n_families_all"] = by_client["family"].nunique()
    out["total_payments"] = by_client["n_payments"].sum()
    out["stream_first_days"] = by_client["days_since_first"].max()
    out["stream_last_days"] = by_client["days_since_last"].min()
    for window in (60, 90):
        out[f"stream_activity_ratio{window}"] = _activity_ratio(payments, cutoff, window).reindex(clients)
    for family in MERCHANT_FAMILIES:
        out[f"active_has_{family}"] = count(active[active["family"] == family]).clip(upper=1)
    return pd.DataFrame(out, index=clients)[FEATURE_COLUMNS].astype(float)


def _stream_fields(streams: pd.DataFrame, payments: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Per stream: its candidate fields and its churn signals from its member payments."""
    s = pd.DataFrame(
        {
            "client_id": streams["client_id"].astype(str).to_numpy(),
            "stream_id": streams["stream_id"].to_numpy(dtype=int),
            "family": streams["family"].astype(str).to_numpy(),
            "active": streams["active"].to_numpy(dtype=bool),
            "n_payments": streams["n_payments"].to_numpy(dtype=float),
            "days_since_first": ((cutoff - streams["first_payment"]) / _DAY).to_numpy(dtype=float),
            "days_since_last": ((cutoff - streams["last_payment"]) / _DAY).to_numpy(dtype=float),
            "period_days": streams["period_days"].to_numpy(dtype=float),
            "gap_mad_days": streams["gap_mad_days"].to_numpy(dtype=float),
            "amount_cv": streams["amount_cv"].to_numpy(dtype=float),
            "amount": streams["median_amount"].to_numpy(dtype=float),
            "refund_rate": streams["refund_rate"].to_numpy(dtype=float),
            "days_to_next": ((streams["next_payment"] - cutoff) / _DAY).to_numpy(dtype=float),
            "n_refunds": streams["n_refunds"].to_numpy(dtype=float),
            "next_dom": streams["next_payment"].dt.day.to_numpy(dtype=float),
        }
    )
    paid = payments.assign(client_id=payments["client_id"].astype(str))
    paid = paid.sort_values(["client_id", "stream_id", "timestamp", "amount"], kind="stable")
    days = ((paid["timestamp"] - cutoff) / _DAY).to_numpy(dtype=float)  # negative: before the Cutoff
    dom = paid["timestamp"].dt.day.to_numpy(dtype=float)
    refunded = paid["refunded"].to_numpy(dtype=bool)
    keys = list(zip(paid["client_id"], paid["stream_id"]))
    starts = np.flatnonzero([True] + [a != b for a, b in zip(keys[1:], keys[:-1])]) if keys else np.zeros(0, int)
    ends = np.append(starts[1:], len(keys)).astype(int)
    position = {keys[a]: (a, b) for a, b in zip(starts, ends)}
    rows = []
    for client, stream, period, gap_mad in zip(s["client_id"], s["stream_id"], s["period_days"], s["gap_mad_days"]):
        a, b = position[(client, stream)]
        rows.append(_churn(days[a:b], refunded[a:b], dom[a:b], period, gap_mad))
    churn = pd.DataFrame(rows, columns=[c for c in CHURN_FIELDS if c not in ("n_refunds", "next_dom")], index=s.index)
    return pd.concat([s, churn.astype(float)], axis=1)


def _churn(days: np.ndarray, refunded: np.ndarray, dom: np.ndarray, period: float, gap_mad: float) -> dict:
    """One stream's churn signals: `days` are its payments' times relative to the Cutoff, in order."""
    n = len(days)
    since_last = -days[-1]
    f = {
        "refund_last": float(refunded[-1]),
        "refund_last2": float(refunded[-2:].any()),
        "refund_last_days": -days[refunded][-1] if refunded.any() else np.nan,
        "last_dom": dom[-1],
        "first_dom": dom[0],
    }
    if n < 2:
        return f
    gaps = np.diff(days)
    folds = np.maximum(1, np.round(gaps / period))
    folded = gaps / folds
    f.update(
        {
            "overdue": since_last / period,
            "last_gap_ratio": gaps[-1] / period,
            "last_gap_missed": float(folds[-1] >= 2),
            "last_gap_late": (folded[-1] - period) / period,
            "n_missed": float((folds - 1).sum()),
            "missed_rate": float((folds - 1).sum() / folds.sum()),
            "missed_last3": float((folds[-3:] - 1).sum()),
            "gap_trend": float(np.polyfit(np.arange(len(gaps)), folded / period, 1)[0]) if len(gaps) >= 3 else np.nan,
            "gap_recent_minus_all": (folded[-3:].mean() - folded.mean()) / period,
            "gap_mad_rel": gap_mad / period,
            "expected_minus_actual": (days[-1] - days[0]) / period + 1 - n,
            "phase": (since_last % period) / period,
        }
    )
    return f


def _activity_ratio(payments: pd.DataFrame, cutoff: pd.Timestamp, window: int) -> pd.Series:
    """Per Client: its stream payments in the last `window` days over its average per `window` days
    before them; unknown for a history not much longer than the window."""
    days = pd.Series(((cutoff - payments["timestamp"]) / _DAY).to_numpy(dtype=float), index=payments["client_id"].astype(str))
    by_client = days.groupby(level=0)
    span = by_client.max()
    recent = (days <= window).groupby(level=0).sum()
    earlier = (days > window).groupby(level=0).sum() / ((span - window) / window)
    ratio = (recent / earlier.where(earlier > 0)).where(span > window + 30)
    return ratio.astype(float)


# --- the model ------------------------------------------------------------------------------


class NoneModel:
    """LightGBM binary model: P(the Client's label is `none`), from `client_features` (plus `RANKER_NONE`)."""

    def __init__(
        self,
        booster: lgb.Booster | None = None,
        constant: float | None = None,
        features: list[str] | None = None,
        fitted_on_clients: list[str] | None = None,
        params: dict | None = None,
    ):
        self.booster = booster
        self.constant = constant  # P(`none`) when training had one outcome only
        self.features = list(NONE_FEATURES if features is None else features)
        # the Clients it was fitted on: real-labelled Clients with at least one Candidate Stream only
        self.fitted_on_clients = list(fitted_on_clients or [])
        self.params = NONE_PARAMS if params is None else params

    def fit(self, x: pd.DataFrame, is_none: pd.Series) -> "NoneModel":
        target = is_none.reindex(x.index).to_numpy(dtype=int)
        self.fitted_on_clients = [str(c) for c in x.index]
        self.booster, self.constant = None, None
        if len(set(target)) < 2:
            self.constant = float(target.mean()) if len(target) else None
            return self
        model = lgb.LGBMClassifier(**self.params)
        model.fit(x[self.features], target)
        self.booster = model.booster_
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """P(`none`) per row of `x`; unknown (NaN) when it was fitted on no Client at all."""
        if self.booster is not None:
            p = np.asarray(self.booster.predict(x[self.features]), dtype=float) if len(x) else np.zeros(0)
        else:
            p = np.full(len(x), np.nan if self.constant is None else self.constant, dtype=float)
        return pd.Series(p, index=x.index, name=NONE_LABEL)

    def to_dict(self) -> dict:
        return {
            "features": self.features,
            "constant": self.constant,
            "fitted_on_clients": self.fitted_on_clients,
            "booster": None if self.booster is None else self.booster.model_to_string(),
        }

    @classmethod
    def from_dict(cls, saved: dict) -> "NoneModel":
        if saved.get("features") != NONE_FEATURES:
            raise ValueError("the ranker's none model was saved with other features; retrain it")
        booster = None if saved["booster"] is None else lgb.Booster(model_str=saved["booster"])
        return cls(booster, saved["constant"], saved["features"], saved["fitted_on_clients"])


def combine(ranker: pd.DataFrame, p_none: pd.Series) -> pd.DataFrame:
    """The ranker's per-Client probabilities (`LABELS`) with `none` from the `none` model: for each
    Client in `p_none`, each family gets its share of the ranker's family probabilities times
    1 - P(`none`). A Client outside `p_none`, with an unknown P(`none`) or without any family score
    keeps the ranker's row. Rows still sum to 1."""
    out = ranker[list(LABELS)].copy()
    p = p_none.reindex(out.index)
    families = out[list(MERCHANT_FAMILIES)]
    total = families.sum(axis=1)
    use = p.notna() & (total > 0)
    out.loc[use, list(MERCHANT_FAMILIES)] = families[use].div(total[use], axis=0).mul(1.0 - p[use], axis=0)
    out.loc[use, NONE_LABEL] = p[use]
    return out
