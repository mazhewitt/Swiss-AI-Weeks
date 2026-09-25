"""Ticket 23: the payments on a Recurring Stream's schedule, and what their descriptions and MCCs say.

Label-free: transactions and the project's stream detector (`streams._detect_per_client`, default
`StreamParams`: ticket 11's Stray Payment join on, ticket 12's Decoy join off) only.

**Payments on a stream's schedule.** For every detected stream of two or more payments (a stream of one
payment has no schedule):

- `member`: every payment the detector put in the stream;
- `slot`: an outgoing card payment before the Cutoff that is in no stream of two or more payments and
  fits an empty slot of the stream, with ticket 11/12's Stray Payment tolerances
  (`streams._join_on_schedule`): the stream has a period of at least `stray_min_period_days` (7); the
  payment's log amount is within `amount_tolerance` (0.06) of the stream's amount range; it is within
  `schedule_tolerance_days` (3) of a whole number of periods from one of the stream's payments, at least
  half a period from all of them, and at most one period (plus the tolerance) before the first or after
  the last. A payment that fits several streams goes to the nearest slot, then the nearest amount.
  These are the noisy payments the detector did not take (Decoy descriptions, shop words, another
  family's words, or a Filler Description that fits no amount/schedule the detector accepted);
- `control`: the same test half a period off the schedule (ticket 11/12's off-phase control), for payments
  that are neither members nor on a slot. It estimates how many `slot` payments are chance hits: the
  control rate per control slot times the number of empty on-phase slots.

**Description kinds**, relative to the stream's family (music and streaming count as one detection group,
as in the detector):

- `own`: the description names the family: one of its family words, or an ambiguous description whose
  families include it ("premium plan" for software, music or streaming; "digital plus" for mobile, music
  or streaming; "monthly plan" for mobile; "service plan" for cloud; these are base descriptions);
- `filler`: names no family and no other family's base description (a Filler Description: "member plan",
  "subscription charge", "digital service", "monthly plan" off mobile, and truncations such as "plan");
- `decoy`: a Decoy description ("digital order", "merchant charge"), never a shop word;
- `other_family`: another group's family words, or an ambiguous base description none of whose families is
  this one ("premium plan" on a gym stream);
- `other`: shop words, fees, or empty.

**MCC noise:** the MCC is not the family's home MCC (`streams.HOME_MCC`; 5812 for music and streaming).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from recurring_family.config import CUTOFF
from recurring_family.streams import (
    HOME_MCC,
    MUSIC_OR_STREAMING,
    StreamParams,
    _AMBIGUOUS,
    _DECOY_WORDS,
    _FAMILY_WORDS,
    _SHOP_WORDS,
    _detect_per_client,
    _words,
)

P = StreamParams()
KINDS = ("own", "filler", "decoy", "other_family", "other")
NOISE_KINDS = KINDS[1:]
HOME = {("music" if v == MUSIC_OR_STREAMING else v): k for k, v in HOME_MCC.items()}
HOME["streaming"] = HOME["music"]
DAY_NS = 86400e9


def group_of(family: str) -> str:
    return MUSIC_OR_STREAMING if family in ("music", "streaming") else family


def kind(description: str, family: str) -> str:
    words = set(_words(description))
    if not words or words & _SHOP_WORDS:
        return "other"
    if words & _DECOY_WORDS:
        return "decoy"
    group = group_of(family)
    named = {g for g, keys in _FAMILY_WORDS.items() if words & keys}
    if group in named:
        return "own"
    if named:
        return "other_family"
    for required, families in _AMBIGUOUS:  # the detector's order: the first pattern that matches
        if required <= words:
            if group in families:
                return "own"
            return "filler" if required == {"monthly", "plan"} else "other_family"
    return "filler"


def home_mcc(family: str) -> str:
    return HOME[family]


def schedule_payments(transactions: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF, params: StreamParams = P):
    """(payments, streams, table) for every Client in `transactions`.

    payments: one row per on-schedule (`member` or `slot`) or `control` payment: client_id, stream key,
    family, role, the transaction's index label in `transactions` (`row`), description, mcc, kind and
    `mcc_noise`.
    streams: one row per stream of two or more payments: client_id, stream key, family, n_payments,
    period_days, missed slots in its gaps, observable empty end slots, on-phase and control slot counts.
    table: the detector's stream table (every stream, including those of one payment).
    """
    pay_rows, stream_rows, table = [], [], []
    first_seen = transactions.groupby("client_id")["timestamp"].min()
    cut = cutoff.value / DAY_NS
    tol, tol_a = params.schedule_tolerance_days, params.amount_tolerance
    for client, pay, srows, membership, _ in _detect_per_client(transactions, cutoff, params):
        table.extend(srows)
        t = pd.DatetimeIndex(pay["timestamp"]).as_unit("ns").asi8 / DAY_NS
        la = np.log(pay["amount"].to_numpy(float).clip(1e-9))
        desc = pay["description"].astype(object).to_numpy()
        mcc = pay["mcc"].astype(object).to_numpy()
        index = pay.index.to_numpy()
        start = pd.Timestamp(first_seen[client]).value / DAY_NS
        long_ids = [r["stream_id"] for r in srows if r["n_payments"] >= 2]
        free = np.flatnonzero(~np.isin(membership, long_ids))
        best_on, best_ctrl = {}, {}  # payment -> (off, amount gap, stream id)
        info = {}
        for r in srows:
            s = r["stream_id"]
            if r["n_payments"] < 2:
                continue
            members = np.flatnonzero(membership == s)
            members = members[np.argsort(t[members], kind="stable")]
            per = float(r["period_days"])
            T = t[members]
            folds = np.maximum(1, np.round(np.diff(T) / per))
            missed = int((folds - 1).sum())
            ends = int(T[0] - per >= start - tol) + int(T[-1] + per < cut)
            info[s] = (r, members)
            stream_rows.append({
                "client_id": client, "stream": f"{client}:{s}", "family": r["family"], "n_payments": len(members),
                "period_days": per, "missed": missed, "folds": int(folds.sum()), "end_slots": ends,
                "on_slots": missed + ends, "control_slots": int(folds.sum()) + ends,
                "active": bool(r["active"]),
            })
            if not per >= params.stray_min_period_days:
                continue
            lo, hi = la[members].min(), la[members].max()
            for j in free:
                gap = max(lo - la[j], la[j] - hi, 0.0)
                if gap > tol_a:
                    continue
                apart = t[j] - T
                near = np.abs(apart).min()
                off = float(np.abs(apart - np.round(apart / per) * per).min())
                if near >= per / 2 and T[0] - per - tol <= t[j] <= T[-1] + per + tol and off <= tol:
                    if j not in best_on or (off, gap) < best_on[j][:2]:
                        best_on[j] = (off, gap, s)
                    continue
                off_c = float(np.abs(apart - (np.floor(apart / per) + 0.5) * per).min())
                if near >= per / 2 - tol and T[0] - per / 2 - tol <= t[j] <= T[-1] + per / 2 + tol and off_c <= tol:
                    if j not in best_ctrl or (off_c, gap) < best_ctrl[j][:2]:
                        best_ctrl[j] = (off_c, gap, s)
        for s, (r, members) in info.items():
            for j in members:
                pay_rows.append((client, f"{client}:{s}", r["family"], "member", index[j], desc[j], mcc[j]))
        for j, (_, _, s) in best_on.items():
            pay_rows.append((client, f"{client}:{s}", info[s][0]["family"], "slot", index[j], desc[j], mcc[j]))
        for j, (_, _, s) in best_ctrl.items():
            if j not in best_on:
                pay_rows.append((client, f"{client}:{s}", info[s][0]["family"], "control", index[j], desc[j], mcc[j]))
    payments = pd.DataFrame(pay_rows, columns=["client_id", "stream", "family", "role", "row", "description", "mcc"])
    payments["kind"] = [kind(d, f) for d, f in zip(payments["description"], payments["family"])]
    payments["mcc_noise"] = [m != home_mcc(f) for m, f in zip(payments["mcc"], payments["family"])]
    return payments, pd.DataFrame(stream_rows), pd.DataFrame(table)


def summarise(payments: pd.DataFrame, streams: pd.DataFrame, table: pd.DataFrame, n_clients: int) -> dict:
    """Noise and fragmentation of one split. Chance `slot` hits are removed per kind (and per family): the
    control hits times on-phase slots over control slots, split-wide."""
    ratio = streams["on_slots"].sum() / max(streams["control_slots"].sum(), 1)
    on = payments[payments["role"] != "control"]
    ctrl = payments[payments["role"] == "control"]

    def shares(on_f: pd.DataFrame, ctrl_f: pd.DataFrame) -> dict:
        counts = on_f.groupby("kind").size().reindex(KINDS, fill_value=0).astype(float)
        slot = on_f[on_f["role"] == "slot"].groupby("kind").size().reindex(KINDS, fill_value=0)
        chance = (ctrl_f.groupby("kind").size().reindex(KINDS, fill_value=0) * ratio).clip(upper=slot)
        corrected = counts - chance
        n = corrected.sum()
        # MCC noise, with chance slot hits removed in proportion to the control's MCC-noise share
        ctrl_mcc = float(ctrl_f["mcc_noise"].mean()) if len(ctrl_f) else 0.0
        mcc_noisy = float(on_f["mcc_noise"].sum()) - float(chance.sum()) * ctrl_mcc
        members = on_f[on_f["role"] == "member"]
        return {
            "n_on_schedule": int(len(on_f)), "n_members": int(len(members)),
            "n_slot": int((on_f["role"] == "slot").sum()), "chance_slot_hits": round(float(chance.sum()), 2),
            "description": {k: round(float(corrected[k] / n), 4) for k in KINDS},
            "description_noise": round(float(1 - corrected["own"] / n), 4),
            "mcc_noise": round(mcc_noisy / n, 4),
            "members_only": {
                "description_noise": round(float((members["kind"] != "own").mean()), 4),
                "mcc_noise": round(float(members["mcc_noise"].mean()), 4),
            },
        }

    out = {"all": shares(on, ctrl)}
    out["by_family"] = {f: shares(on[on["family"] == f], ctrl[ctrl["family"] == f]) for f in sorted(on["family"].unique())}
    slot = on[on["role"] == "slot"]
    out["slot_by_kind"] = {k: int((slot["kind"] == k).sum()) for k in KINDS}
    out["control_by_kind"] = {k: int((ctrl["kind"] == k).sum()) for k in KINDS}
    out["on_over_control_slots"] = round(float(ratio), 4)
    # independence of the two noises, on members and slot payments
    noisy_desc = on["kind"] != "own"
    out["mcc_noise_given_description"] = {
        "own": round(float(on.loc[~noisy_desc, "mcc_noise"].mean()), 4),
        "not_own": round(float(on.loc[noisy_desc, "mcc_noise"].mean()), 4),
    }
    # per-stream overdispersion of description noise (1 = independent per payment)
    per = on.groupby("stream").agg(n=("kind", "size"), k=("kind", lambda s: int((s != "own").sum())))
    p = per["k"].sum() / per["n"].sum()
    expected = (per["n"] * p * (1 - p)).sum()
    observed = ((per["k"] - per["n"] * p) ** 2).sum()
    out["description_noise_dispersion"] = round(float(observed / expected), 3) if expected else None
    n_long = len(streams)
    out["fragmentation"] = {
        "clients": n_clients,
        "streams_per_client": round(len(table) / n_clients, 4),
        "streams_2plus_per_client": round(n_long / n_clients, 4),
        "single_payment_stream_share": round(float((table["n_payments"] < 2).mean()), 4),
        "payments_per_stream_2plus": round(float(streams["n_payments"].mean()), 3),
        "missed_slots_per_stream": round(float(streams["missed"].mean()), 4),
        "missed_slot_rate": round(float(streams["missed"].sum() / streams["folds"].sum()), 4),
        "active_streams_per_client": round(float(table["active"].sum() / n_clients), 4),
        "stream_payments_per_client": round(float(table["n_payments"].sum() / n_clients), 3),
    }
    return out
