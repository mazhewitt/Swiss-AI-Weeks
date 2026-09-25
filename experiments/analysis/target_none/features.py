"""Ticket 18 feature blocks, per Client, from history before the Cutoff. Label-free: every function here
takes transactions only.

- base: the v2 `none` model's stream features (`none_model.FEATURE_COLUMNS`, from the stream table and its
  member payments only).
- raw: ticket 16's kitchen sink without the description-word bags (all-history, last-90-day and last-payment
  word bags are dropped; the 25 MCC bags stay), plus Decoy counts and their activity-normalised last-90-day
  share, Decoys within 6% of an Active Stream's amount, currency / hour / MCC mix, and refund counts and timing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/analysis/none_probe")
from kitchen_sink_none import raw_features as kitchen_sink  # noqa: E402
from recurring_family.config import CUTOFF  # noqa: E402
from recurring_family.none_model import FEATURE_COLUMNS as BASE_COLUMNS, client_features  # noqa: E402
from recurring_family.ranker import _detected  # noqa: E402
from recurring_family.streams import StreamParams, _decoy_described  # noqa: E402

DAY = pd.Timedelta(days=1)
CURRENCIES = ("chf", "eur", "usd", "gbp")
TYPES = ("card_payment", "topup", "p2p_transfer", "transfer", "atm", "refund", "fee")
HOUR_BINS = {"night": range(0, 6), "early": range(6, 8), "day": range(8, 22), "late": range(22, 24)}
DECOY_TOLERANCE = StreamParams().amount_tolerance  # 0.06 log-amount: the detector's own "within 6%"


def before_cutoff(tx: pd.DataFrame) -> pd.DataFrame:
    return tx[tx["timestamp"] < CUTOFF]


def base_features(tx: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
    streams, paid = _detected(tx, clients)
    return client_features(streams, paid).reindex(clients)[BASE_COLUMNS].astype(float)


def _entropy(shares: pd.DataFrame) -> pd.Series:
    p = shares.clip(lower=1e-12)
    return -(shares * np.log(p)).sum(axis=1)


def _mix(tx: pd.DataFrame, column: str, values, prefix: str, clients: pd.Index) -> pd.DataFrame:
    counts = pd.crosstab(tx["client_id"], tx[column]).reindex(index=clients, columns=list(values)).fillna(0)
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    shares.columns = [f"{prefix}_share_{v}" for v in values]
    return shares


def extra_features(tx: pd.DataFrame, clients: pd.Index, mccs: list[str]) -> pd.DataFrame:
    streams, _ = _detected(tx, clients)
    tx = tx.copy()
    tx["days"] = (CUTOFF - tx["timestamp"]) / DAY
    tx["card"] = tx["type"] == "card_payment"
    tx["decoy"] = tx["description"].map(_decoy_described) & tx["card"]
    tx["hour"] = tx["timestamp"].dt.hour
    tx["hour_bin"] = tx["hour"].map({h: b for b, hs in HOUR_BINS.items() for h in hs})
    f = pd.DataFrame(index=clients)
    g = tx.groupby("client_id")
    n_tx = g.size().reindex(clients).fillna(0)
    n_tx90 = tx[tx["days"] <= 90].groupby("client_id").size().reindex(clients).fillna(0)
    activity90 = n_tx90 / n_tx.replace(0, np.nan)  # the Client's own last-90-day share of activity

    # Decoys: counts, share of card payments, last-90-day share normalised by the Client's own activity
    dec = tx[tx["decoy"]]
    n_dec = dec.groupby("client_id").size().reindex(clients).fillna(0)
    n_dec90 = dec[dec["days"] <= 90].groupby("client_id").size().reindex(clients).fillna(0)
    n_card = tx[tx["card"]].groupby("client_id").size().reindex(clients).fillna(0)
    f["decoy_n"] = n_dec
    f["decoy_share_of_card"] = n_dec / n_card.replace(0, np.nan)
    f["decoy_share90"] = n_dec90 / n_dec.replace(0, np.nan)
    f["decoy_share90_norm"] = f["decoy_share90"] / activity90.replace(0, np.nan)
    f["decoy_rate90_minus_before"] = n_dec90 / 90 - (n_dec - n_dec90) / (g["days"].max().reindex(clients) - 90).clip(lower=1)

    # Decoys within 6% (log-amount) of an Active Stream's median amount
    active = streams[streams["active"]][["client_id", "median_amount"]].astype({"client_id": str})
    d = dec[["client_id", "amount", "days"]].reset_index().astype({"client_id": str})
    near = d.merge(active, on="client_id", how="inner")
    near = near[np.abs(np.log(near["amount"] / near["median_amount"])) <= DECOY_TOLERANCE]
    near = d[d["index"].isin(set(near["index"]))]
    f["decoy_near_active_n"] = near.groupby("client_id").size().reindex(clients.astype(str)).fillna(0).to_numpy()
    f["decoy_near_active_n90"] = near[near["days"] <= 90].groupby("client_id").size().reindex(clients.astype(str)).fillna(0).to_numpy()
    f["decoy_near_active_share"] = f["decoy_near_active_n"] / n_dec.replace(0, np.nan)

    # currency, hour, type and MCC mix
    cur = _mix(tx, "currency", CURRENCIES, "cur", clients)
    card_cur = _mix(tx[tx["card"]], "currency", CURRENCIES, "cardcur", clients)
    hours = _mix(tx, "hour_bin", HOUR_BINS, "hour", clients)
    types = _mix(tx, "type", TYPES, "type", clients)
    card = tx[tx["card"]]
    mcc = _mix(card[card["mcc"].isin(mccs)], "mcc", mccs, "mcc", clients)
    f["cur_n"] = g["currency"].nunique().reindex(clients)
    f["cur_entropy"] = _entropy(cur.fillna(0))
    f["hour_mean"] = g["hour"].mean().reindex(clients)
    f["hour_std"] = g["hour"].std().reindex(clients)
    f["card_hour_mean"] = card.groupby("client_id")["hour"].mean().reindex(clients)
    f["mcc_entropy"] = _entropy(mcc.fillna(0))
    f["mcc_top_share"] = mcc.max(axis=1)
    f["mcc_other_share"] = 1 - card[card["mcc"].isin(mccs)].groupby("client_id").size().reindex(clients) / n_card.replace(0, np.nan)

    # refunds: counts, windows, timing
    ref = tx[tx["type"] == "refund"]
    rg = ref.groupby("client_id")
    n_ref = rg.size().reindex(clients).fillna(0)
    f["refund_type_n"] = n_ref
    f["refund_type_share_of_card"] = n_ref / n_card.replace(0, np.nan)
    f["refund_type_n90"] = ref[ref["days"] <= 90].groupby("client_id").size().reindex(clients).fillna(0)
    f["refund_type_share90_norm"] = (f["refund_type_n90"] / n_ref.replace(0, np.nan)) / activity90.replace(0, np.nan)
    f["refund_type_first_days"] = rg["days"].max().reindex(clients)
    f["refund_type_last_days"] = rg["days"].min().reindex(clients)
    f["refund_type_gap_median"] = ref.sort_values("days").groupby("client_id")["days"].apply(lambda s: s.diff().median()).reindex(clients)
    f["refund_type_amount_mean"] = rg["amount"].mean().reindex(clients)
    # days from each refund back to the nearest earlier card payment of the same amount (a matched reversal)
    lag = _refund_lag(card, ref)
    f["refund_matched_share"] = lag.groupby("client_id")["matched"].mean().reindex(clients)
    f["refund_lag_median"] = lag.groupby("client_id")["lag"].median().reindex(clients)
    return pd.concat([f, cur, card_cur, hours, types], axis=1).astype(float)


def _refund_lag(card: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    c = card[["client_id", "amount", "timestamp"]].assign(key=lambda d: d["client_id"].astype(str) + "|" + d["amount"].round(2).astype(str))
    r = ref[["client_id", "amount", "timestamp"]].assign(key=lambda d: d["client_id"].astype(str) + "|" + d["amount"].round(2).astype(str))
    r = r.sort_values("timestamp").reset_index(drop=True)
    c = c.sort_values("timestamp").rename(columns={"timestamp": "paid_at"})[["key", "paid_at"]]
    m = pd.merge_asof(r, c, left_on="timestamp", right_on="paid_at", by="key", direction="backward")
    m["lag"] = (m["timestamp"] - m["paid_at"]) / DAY
    m["matched"] = m["lag"].notna().astype(float)
    return m


def raw_block(tx: pd.DataFrame, clients: pd.Index, mccs: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    sink, vocab, mccs = kitchen_sink(tx, mccs=mccs)
    bags = {f"{p}_{v}" for v in vocab for p in ("w", "w90", "last")}
    sink = sink.drop(columns=[c for c in sink.columns if c in bags]).reindex(clients)
    extra = extra_features(tx, clients, mccs)
    return pd.concat([sink, extra], axis=1).replace([np.inf, -np.inf], np.nan).astype(float), mccs


def blocks(tx: pd.DataFrame, clients: pd.Index, mccs: list[str] | None = None, cache: Path | None = None):
    """(base, raw, mccs) for `clients`, cached as pickles under `cache` when given."""
    if cache is not None and (cache / "base.pkl").exists():
        return pd.read_pickle(cache / "base.pkl"), pd.read_pickle(cache / "raw.pkl"), pd.read_pickle(cache / "mccs.pkl")
    tx = before_cutoff(tx)
    base = base_features(tx, clients)
    raw, mccs = raw_block(tx, clients, mccs)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        base.to_pickle(cache / "base.pkl"); raw.to_pickle(cache / "raw.pkl"); pd.to_pickle(mccs, cache / "mccs.pkl")
    return base, raw, mccs
