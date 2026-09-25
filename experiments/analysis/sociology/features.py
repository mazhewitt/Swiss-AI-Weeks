"""Ticket 20: the six sociological churn features, per Client, from history before the Cutoff (2026-01-01).

Fixed in code, and committed, before any label was read. Every function here takes transactions only.

Schema mapping (from a label-free look at the train transactions; see findings.md):

- Inbound top-up / salary: `type == "topup"` (always `direction == "in"`, description always "salary").
  Received P2P transfers are not top-ups or salary and are left out.
- Outgoing card spend: `type == "card_payment"` and `direction == "out"` (every card payment is outgoing).
  All card payments count, recurring ones included. Amounts are in each transaction's own currency; the
  features are within-Client ratios, so no conversion is made.
- Active Streams, member payments and "no longer active": the existing detector's stream table
  (`recurring_family.streams`, through the ranker's memoised `_detected`), with its default parameters.

The six features (the sociological prediction for `none` in brackets):

1. income_trend        log((topup amount in the last 90 days + 1) / (mean 90-day topup amount over the
                       270 days before that + 1))                                         [lower -> none]
2. spend_contraction   the same log ratio for outgoing card spend; negative = shrinking   [lower -> none]
3. essential_share     share of card spend (amount) in the last 180 days at ESSENTIAL_MCCS [higher -> none]
4. price_rise          max over Active Streams of (last amount / median of the stream's earlier amounts - 1),
                       clipped at 0; 0 when the Client has no Active Stream                [higher -> none]
5. discretionary_share share of the Client's Active Streams whose family is streaming, music or gym;
                       0 when the Client has no Active Stream                              [higher -> none]
6. recent_stops        number of the Client's streams whose last payment fell in the 180 days before the
                       Cutoff and that are not active at the Cutoff, counting only streams with at least the
                       detector's `min_payments` payments (a stream that could have been active, so "no
                       longer active" means it lapsed rather than never qualified)           [higher -> none]

Windows: "last 90 days" is [Cutoff - 90d, Cutoff); "the 270 days before that" is [Cutoff - 360d, Cutoff - 90d),
whose mean 90-day amount is its total / 3. "Last 180 days" is [Cutoff - 180d, Cutoff).
"""
import numpy as np
import pandas as pd

from recurring_family.config import CUTOFF
from recurring_family.ranker import _detected
from recurring_family.streams import StreamParams

DAY = pd.Timedelta(days=1)
FEATURES = ["income_trend", "spend_contraction", "essential_share", "price_rise", "discretionary_share", "recent_stops"]

# The sign the churn literature predicts for each feature's association with `none`.
PREDICTED_SIGN = {"income_trend": -1, "spend_contraction": -1, "essential_share": +1, "price_rise": +1,
                  "discretionary_share": +1, "recent_stops": +1}

# Essential MCCs, from the standard ISO 18245 / card-network MCC meanings, fixed before any label was read and
# never adjusted afterwards. Only the categories the ticket names: groceries and supermarkets, pharmacies,
# utilities, fuel, public transport. 4814 (telecommunication services) is not "utilities" in the standard
# meaning (that is 4900) and is left out.
ESSENTIAL_MCCS = {
    # groceries, supermarkets, food stores
    "5411": "grocery stores, supermarkets",
    "5422": "freezer and locker meat provisioners",
    "5441": "candy, nut and confectionery stores",
    "5451": "dairy products stores",
    "5462": "bakeries",
    "5499": "misc. food stores, convenience stores",
    # pharmacies
    "5912": "drug stores and pharmacies",
    "5122": "drugs, drug proprietaries and druggist sundries",
    # utilities
    "4900": "utilities: electric, gas, water, sanitary",
    # fuel
    "5541": "service stations",
    "5542": "automated fuel dispensers",
    "5983": "fuel dealers",
    # public transport
    "4111": "local and suburban commuter passenger transportation, incl. ferries",
    "4112": "passenger railways",
    "4131": "bus lines",
}
DISCRETIONARY_FAMILIES = ("streaming", "music", "gym")


def _window_sum(tx: pd.DataFrame, clients: pd.Index, lo_days: float, hi_days: float) -> pd.Series:
    """Per Client, the amount summed over transactions in [Cutoff - hi_days, Cutoff - lo_days)."""
    days = (CUTOFF - tx["timestamp"]) / DAY
    w = tx[(days > lo_days) & (days <= hi_days)]
    return w.groupby("client_id")["amount"].sum().reindex(clients).fillna(0.0)


def _log_trend(tx: pd.DataFrame, clients: pd.Index) -> pd.Series:
    last90 = _window_sum(tx, clients, 0, 90)
    before = _window_sum(tx, clients, 90, 360) / 3.0  # mean 90-day amount over the 270 days before
    return np.log((last90 + 1.0) / (before + 1.0))


def sociology_features(tx: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
    """The six features for `clients` (index of string client ids), from transactions before the Cutoff."""
    clients = pd.Index(clients.astype(str), name="client_id")
    tx = tx.assign(client_id=tx["client_id"].astype(str))
    tx = tx[(tx["timestamp"] < CUTOFF) & tx["client_id"].isin(set(clients))]
    f = pd.DataFrame(index=clients)

    topup = tx[(tx["type"] == "topup") & (tx["direction"] == "in")]
    card = tx[(tx["type"] == "card_payment") & (tx["direction"] == "out")]
    f["income_trend"] = _log_trend(topup, clients)
    f["spend_contraction"] = _log_trend(card, clients)

    card180 = card[(CUTOFF - card["timestamp"]) / DAY <= 180]
    total = card180.groupby("client_id")["amount"].sum().reindex(clients)
    essential = card180[card180["mcc"].isin(set(ESSENTIAL_MCCS))].groupby("client_id")["amount"].sum().reindex(clients).fillna(0.0)
    f["essential_share"] = (essential / total.replace(0, np.nan)).fillna(0.0)

    streams, paid = _detected(tx, clients)
    streams = streams.assign(client_id=streams["client_id"].astype(str))
    paid = paid.assign(client_id=paid["client_id"].astype(str))
    active = streams[streams["active"]]

    # price rise: last member payment against the median of the stream's earlier member payments
    key = ["client_id", "stream_id"]
    ap = paid.merge(active[key], on=key, how="inner").sort_values(key + ["timestamp"], kind="stable")
    rise = ap.groupby(key, sort=False)["amount"].apply(lambda a: a.iloc[-1] / np.median(a.iloc[:-1].to_numpy()) - 1.0 if len(a) > 1 else np.nan)
    rise = rise.clip(lower=0).groupby(level="client_id").max()
    f["price_rise"] = rise.reindex(clients).fillna(0.0)

    n_active = active.groupby("client_id").size().reindex(clients).fillna(0)
    n_disc = active[active["family"].isin(DISCRETIONARY_FAMILIES)].groupby("client_id").size().reindex(clients).fillna(0)
    f["discretionary_share"] = (n_disc / n_active.replace(0, np.nan)).fillna(0.0)

    since_last = (CUTOFF - streams["last_payment"]) / DAY
    stopped = streams[(~streams["active"]) & (since_last <= 180) & (streams["n_payments"] >= StreamParams().min_payments)]
    f["recent_stops"] = stopped.groupby("client_id").size().reindex(clients).fillna(0).astype(float)
    return f[FEATURES].astype(float)
