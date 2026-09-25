"""Stream detection: Client transactions + Cutoff -> one row per Recurring Stream.

Only outgoing card payments before the Cutoff are candidates. Each candidate gets
Merchant Family evidence from its description and MCC:

- a *family keyword* ("gym", "saas", "audio", ...) names one family. On that family's home MCC it
  starts a stream; on any other MCC it only joins an existing stream of the family, like a filler;
- an *ambiguous* description ("premium plan", "digital plus", "monthly plan") names a set
  of families, and the MCCs of its amount cluster pick one;
- a *Filler Description* ("member plan", "subscription charge", ...) names none. It never
  starts a stream but joins one whose amount and MCC it fits.

A payment whose description names no family and that none of the above placed (a Filler Description
on another family's home MCC or on no home MCC, or an ambiguous description no family of it fits) is
a *stray*: it joins a stream whose amount it fits when its date also fits an empty slot of that
stream's schedule, and never starts one. Valid and test book many stream payments this way. With
`StreamParams.join_decoys`, a payment with a Decoy description joins under the same rules, after the
strays: valid and test also book some stream payments that way.

Shop descriptions, service fees and Decoy Transactions ("digital order", "merchant charge",
...) are otherwise dropped. Music and streaming share MCC 5812: a description hint ("audio", "video", ...)
puts a payment in its family before amount clustering, so two streams at nearby amounts never
chain into one. Unhinted payments ("premium plan") join the hinted stream whose amount they fit;
the rest form streams that are split into music or streaming by amount.

A refund is credited to a stream only when it reverses one of the stream's payments: a payment
of matching amount made shortly before it. Each payment is reversed at most once.
`detect_stream_payments` also returns every stream's member payments, each marked when a matched
refund reverses it.

Each stream also reports its most common MCC and description (words lower-cased, abbreviations
expanded) and the share of its payments whose description is family-specific: a family keyword
names exactly one family, unlike an ambiguous description or a Filler Description.

`pseudo_labels` gives each Client's Pseudo-Label at a Shifted Cutoff: the Merchant Family of the
first Recurring Stream payment in its Horizon, or `none`. Streams for it are detected over the whole
known history, so a stream that starts inside the Horizon counts once it has repeated. Its settings
(`LabellerParams`) can make the task more like the real one: a minimum number of payments a stream
needs before the Shifted Cutoff, and a share of Clients whose streams all stop at the Shifted Cutoff,
as streams stop at the real Cutoff far more often than anywhere inside the known history.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .config import CUTOFF, HISTORY_END, HORIZON, LABEL_COLUMN, NONE_LABEL, SHIFTED_CUTOFF

MUSIC_OR_STREAMING = "music_or_streaming"  # detection group for MCC 5812 before the split

HOME_MCC = {
    "5732": "cloud",
    "7997": "gym",
    "6300": "insurance",
    "4814": "mobile",
    "5734": "software",
    "5812": MUSIC_OR_STREAMING,
}

_ABBREVIATIONS = {"prem": "premium", "dgtl": "digital", "mth": "monthly", "prod": "productivity"}

_SHOP_WORDS = frozenset(
    "shop market foods fresh grocery store neighborhood marketplace hotel booking pharmacy ride share "
    "coffee casual dining electronics salary atm withdrawal p2p send receive fee".split()
)
_DECOY_WORDS = frozenset({"order", "merchant", "purchase", "payment"})

# Family keywords: each names exactly one detection group. The hint splits music from streaming.
_FAMILY_WORDS = {
    "cloud": {"cloud", "storage", "backup"},
    "gym": {"gym", "fit", "fitness", "club", "urban", "membership"},
    "insurance": {"cover", "insurance", "safe", "policy"},
    "mobile": {"phone", "contract", "bill"},
    "software": {"saas", "productivity", "suite", "software"},
    MUSIC_OR_STREAMING: {"audio", "pass", "media", "video", "streaming", "stream"},
}
_MUSIC_HINTS = {"audio", "pass"}
_STREAMING_HINTS = {"media", "video"}

# Ambiguous descriptions: (required words, families they may belong to).
_AMBIGUOUS = (
    ({"premium"}, frozenset({"software", MUSIC_OR_STREAMING})),
    ({"digital", "plus"}, frozenset({"mobile", MUSIC_OR_STREAMING})),
    ({"monthly", "plan"}, frozenset({"mobile"})),  # off 4814 it falls back to filler for the home family
    ({"service", "plan"}, frozenset({"cloud"})),  # last: "service" is also a noise suffix
)

COLUMNS = [
    "client_id",
    "stream_id",
    "family",
    "period_days",
    "median_amount",
    "amount_cv",
    "gap_mad_days",
    "n_payments",
    "first_payment",
    "last_payment",
    "next_payment",
    "active",
    "n_refunds",
    "refund_rate",
    "mcc",
    "description",
    "family_description_share",
]

PAYMENT_COLUMNS = ["client_id", "stream_id", "timestamp", "amount", "refunded"]


@dataclass(frozen=True)
class StreamParams:
    """Every detector threshold, in one place so they can be swept."""

    amount_tolerance: float = 0.06  # log-amount gap that splits two amount clusters
    active_periods: float = 1.6  # Active Stream: last payment within this many periods of the Cutoff
    min_payments: int = 3  # Active Stream: at least this many payments
    canonical_periods: tuple[float, ...] = (14.0, 30.4)  # biweekly and monthly, in days
    music_streaming_split: float = 15.5  # 5812 streams without a hint: below is music, above streaming
    refund_window_days: float = 7.0  # a refund reverses a payment made at most this many days before it
    schedule_tolerance_days: float = 3.0  # a stray payment joins a stream within this many days of an empty slot
    stray_min_period_days: float = 7.0  # only a stream whose period is at least this many days takes stray payments
    join_strays: bool = True  # False: stray payments join no stream (the detector before ticket 11)
    join_decoys: bool = False  # True: a Decoy-described payment joins like a stray one (needs join_strays; ticket 12)


# --- per-row evidence ---------------------------------------------------------


def _words(description: str) -> list[str]:
    return [_ABBREVIATIONS.get(w, w) for w in re.findall(r"[a-z0-9]+", str(description).lower())]


def _canonical_description(description: str) -> str:
    return " ".join(_words(description))


def _family_specific(description: str) -> bool:
    """A family keyword names exactly one detection group (music and streaming count as one)."""
    words = set(_words(description))
    return sum(bool(words & keys) for keys in _FAMILY_WORDS.values()) == 1


def _evidence(description: str, mcc: str) -> tuple[str, frozenset[str], str | None]:
    """(kind, families, hint): kind is 'family', 'ambiguous', 'filler', 'stray' (a description that
    names no family, off every home MCC) or 'drop'."""
    words = set(_words(description))
    if not words or words & _SHOP_WORDS or words & _DECOY_WORDS:
        return "drop", frozenset(), None
    hint = "music" if words & _MUSIC_HINTS else "streaming" if words & _STREAMING_HINTS else None
    named = frozenset(g for g, keys in _FAMILY_WORDS.items() if words & keys)
    if len(named) == 1:
        return ("family" if HOME_MCC.get(mcc) in named else "filler"), named, hint
    if len(named) > 1:
        return "filler", named, None
    for required, families in _AMBIGUOUS:
        if required <= words:
            return "ambiguous", families, hint
    home = HOME_MCC.get(mcc)
    return ("filler", frozenset({home}), None) if home else ("stray", frozenset(), None)


def _names_family(description: str) -> bool:
    words = set(_words(description))
    return any(words & keys for keys in _FAMILY_WORDS.values())


def _decoy_described(description: str) -> bool:
    """A Decoy description ("digital order", "merchant charge", ...), never a shop payment or fee."""
    words = set(_words(description))
    return bool(words & _DECOY_WORDS) and not words & _SHOP_WORDS


def _group_choices(families, hint) -> list[set[str]]:
    """The detection groups a payment of `families` may join, in order of preference. Music or
    streaming narrows to the description hint first, then falls back to either: a hint on a
    stray-MCC payment is occasionally wrong."""
    if MUSIC_OR_STREAMING not in families:
        return [set(families)]
    either = set(families) | {"music", "streaming"}
    return [set(families) | {hint}, either] if hint else [either]


def _amount_clusters(log_amounts: np.ndarray, tolerance: float) -> np.ndarray:
    """Cluster ids by sorted log amount, splitting where neighbours differ by more than `tolerance`."""
    order = np.argsort(log_amounts, kind="stable")
    breaks = np.concatenate([[0], np.cumsum(np.diff(log_amounts[order]) > tolerance)])
    ids = np.empty(len(order), dtype=int)
    ids[order] = breaks
    return ids


# --- detection ----------------------------------------------------------------


def detect_streams(
    transactions: pd.DataFrame,
    cutoff: pd.Timestamp = CUTOFF,
    params: StreamParams = StreamParams(),
) -> pd.DataFrame:
    """One row per Recurring Stream, for every Client in `transactions` (columns: `COLUMNS`)."""
    return detect_stream_payments(transactions, cutoff, params)[0]


def detect_stream_payments(
    transactions: pd.DataFrame,
    cutoff: pd.Timestamp = CUTOFF,
    params: StreamParams = StreamParams(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The stream table of `detect_streams`, and its streams' member payments: one row per payment that
    belongs to a Recurring Stream (columns: `PAYMENT_COLUMNS`), ordered by Client, stream, time and
    amount. `refunded` marks a payment that a refund credited to its stream reverses."""
    rows, paid = [], []
    for client, payments, client_rows, membership, reversed_ in _detect_per_client(transactions, cutoff, params):
        rows.extend(client_rows)
        members = np.flatnonzero(membership >= 0)
        paid.append(
            pd.DataFrame(
                {
                    "client_id": client,
                    "stream_id": membership[members],
                    "timestamp": payments["timestamp"].to_numpy()[members],
                    "amount": payments["amount"].to_numpy(dtype=float)[members],
                    "refunded": reversed_[members],
                }
            )
        )
    return _stream_table(rows), _payment_table(paid)


def _payment_table(parts: list[pd.DataFrame]) -> pd.DataFrame:
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PAYMENT_COLUMNS)
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True)
    table = table.astype({"client_id": "string", "stream_id": "int64", "amount": "float64", "refunded": "bool"})
    return (
        table[PAYMENT_COLUMNS]
        .sort_values(["client_id", "stream_id", "timestamp", "amount"], kind="stable")
        .reset_index(drop=True)
    )


def _stream_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=COLUMNS)
    for column in ("first_payment", "last_payment", "next_payment"):
        # a column of NaT only (every stream a single payment) is inferred timezone-naive
        table[column] = pd.to_datetime(table[column], utc=True)
    return table.astype(
        {
            "client_id": "string",
            "stream_id": "int64",
            "family": "string",
            "period_days": "float64",
            "median_amount": "float64",
            "amount_cv": "float64",
            "gap_mad_days": "float64",
            "n_payments": "int64",
            "first_payment": "datetime64[ns, UTC]",
            "last_payment": "datetime64[ns, UTC]",
            "next_payment": "datetime64[ns, UTC]",
            "active": "bool",
            "n_refunds": "int64",
            "refund_rate": "float64",
            "mcc": "string",
            "description": "string",
            "family_description_share": "float64",
        }
    )


def _detect_per_client(transactions, cutoff, params):
    """Per Client with a payment before `cutoff`: (client, payments, stream rows, membership, reversed),
    where membership gives each payment's `stream_id`, or -1 for a payment in no stream, and reversed
    marks each payment that a refund credited to its stream reverses."""
    tx = transactions[transactions["timestamp"] < cutoff]
    payments = tx[(tx["type"] == "card_payment") & (tx["direction"] == "out")]
    refunds = tx[(tx["type"] == "refund") & (tx["direction"] == "in")]
    for client, group in payments.groupby("client_id", sort=True):
        client_refunds = refunds[refunds["client_id"] == client]
        rows, membership, reversed_ = _client_streams(str(client), group, client_refunds, cutoff, params)
        yield str(client), group, rows, membership, reversed_


def _client_streams(client, payments, refunds, cutoff, params):
    ev = [_evidence(d, m) for d, m in zip(payments["description"], payments["mcc"])]
    kind = np.array([e[0] for e in ev], dtype=object)
    families = [e[1] for e in ev]
    hints = [e[2] for e in ev]
    log_amount = np.log(payments["amount"].to_numpy(dtype=float).clip(min=1e-9))
    mcc = payments["mcc"].astype(object).to_numpy()
    group = np.array([next(iter(f)) if k == "family" else None for k, f in zip(kind, families)], dtype=object)

    # Ambiguous descriptions: each amount cluster takes the family its MCCs vote for.
    for fams in {f for k, f in zip(kind, families) if k == "ambiguous"}:
        idx = np.flatnonzero([k == "ambiguous" and f == fams for k, f in zip(kind, families)])
        cid = _amount_clusters(log_amount[idx], params.amount_tolerance)
        for c in np.unique(cid):
            members = idx[cid == c]
            votes = pd.Series([HOME_MCC.get(mcc[i]) for i in members]).dropna()
            votes = votes[votes.isin(fams)]
            if len(votes):
                counts = votes.value_counts()
                group[members] = sorted(counts[counts == counts.max()].index)[0]

    # Music and streaming share MCC 5812: a description hint moves a payment into its family's group
    # before amount clustering, so a music and a streaming stream at nearby amounts never chain.
    for i in np.flatnonzero(group == MUSIC_OR_STREAMING):
        if hints[i]:
            group[i] = hints[i]

    # Anchored payments (a family keyword, or a resolved ambiguous description) form streams.
    stream_of = np.full(len(kind), -1)
    streams = []  # (group, low, high) of log amount

    def add_streams(idx, g):
        cid = _amount_clusters(log_amount[idx], params.amount_tolerance)
        for c in np.unique(cid):
            members = idx[cid == c]
            stream_of[members] = len(streams)
            streams.append((g, log_amount[members].min(), log_amount[members].max()))

    for g in sorted({x for x in group if x is not None and x != MUSIC_OR_STREAMING}):
        add_streams(np.flatnonzero(group == g), g)

    def distance(s, lo, hi):
        _, s_lo, s_hi = streams[s]
        return max(s_lo - hi, lo - s_hi, 0.0)

    def fit(log_a, choices):
        for allowed in choices:
            fitting = [s for s, (g, _, _) in enumerate(streams) if g in allowed]
            fitting = [(distance(s, log_a, log_a), s) for s in fitting]
            fitting = [(d, s) for d, s in fitting if d <= params.amount_tolerance]
            if fitting:
                return min(fitting)[1]
        return -1

    # Unhinted music-or-streaming payments ("premium plan" on 5812) cluster by amount. A cluster that
    # fits hinted streams of one family joins them, merging those it bridges (it carries price drift);
    # one that fits both families splits between them by amount; the rest form their own streams,
    # split into music or streaming by amount below.
    unhinted = np.flatnonzero(group == MUSIC_OR_STREAMING)
    cid = _amount_clusters(log_amount[unhinted], params.amount_tolerance)
    for c in np.unique(cid):
        members = unhinted[cid == c]
        lo, hi = log_amount[members].min(), log_amount[members].max()
        near = [
            s
            for s, (g, _, _) in enumerate(streams)
            if g in ("music", "streaming") and distance(s, lo, hi) <= params.amount_tolerance
        ]
        if len({streams[s][0] for s in near}) == 1:
            for s in near[1:]:
                stream_of[stream_of == s] = near[0]
                streams[s] = (None, math.inf, -math.inf)  # merged away
            stream_of[members] = near[0]
        elif near:
            stream_of[members] = [fit(log_amount[i], [{"music", "streaming"}]) for i in members]
        if len(alone := members[stream_of[members] < 0]):
            add_streams(alone, MUSIC_OR_STREAMING)
        for s in near:  # joined payments widen a stream's amount range
            if streams[s][0] is not None:
                joined = log_amount[stream_of == s]
                streams[s] = (streams[s][0], joined.min(), joined.max())

    # Filler and unresolved rows join the stream whose amount and family they fit; else dropped.
    for i in np.flatnonzero(group == None):  # noqa: E711
        if kind[i] == "filler":
            stream_of[i] = fit(log_amount[i], _group_choices(families[i], hints[i]))
        elif kind[i] == "ambiguous":
            # MCC vote failed: join a fitting stream of a family the description can mean (a stray
            # MCC) or of the MCC's home family (the description is filler there); else no stream.
            home = HOME_MCC.get(mcc[i])
            stream_of[i] = fit(log_amount[i], _group_choices(families[i] | ({home} if home else set()), hints[i]))

    # Stray payments: a description that names no family (a Filler Description or an ambiguous one,
    # never a shop, fee or Decoy Transaction) left out above, often because a stream's payment was
    # booked on another family's home MCC or on no home MCC. It joins a stream whose amount it fits
    # when its date also fits that stream's schedule; it never starts one.
    ns = pd.DatetimeIndex(payments["timestamp"]).as_unit("ns").asi8
    description = payments["description"].astype(object).to_numpy()
    stray = [
        i
        for i in np.flatnonzero(stream_of < 0)
        if kind[i] in ("stray", "ambiguous") or (kind[i] == "filler" and not _names_family(description[i]))
    ]
    # With `join_decoys`, a payment with a Decoy description (never a shop payment or fee) then joins
    # under the same rules, after the strays have taken their slots: valid and test book some stream
    # payments that way. A random Decoy Transaction joins only when it lands on an empty slot at the
    # stream's amount.
    decoys = []
    if params.join_decoys:
        decoys = [i for i in np.flatnonzero(stream_of < 0) if _decoy_described(description[i])]
    in_order = lambda i: (ns[i], log_amount[i], str(description[i]), str(mcc[i]))  # never row order
    ranges = [None if g is None else (lo, hi) for g, lo, hi in streams]
    days = ns / pd.Timedelta(days=1).value
    if stray and params.join_strays:
        _join_on_schedule(sorted(stray, key=in_order), stream_of, ranges, days, log_amount, params)
    if decoys and params.join_strays:
        _join_on_schedule(sorted(stray + decoys, key=in_order), stream_of, ranges, days, log_amount, params)

    refund_counts, reversed_ = _match_refunds(refunds, stream_of, [g for g, _, _ in streams], ns, log_amount, params)

    times = payments["timestamp"].to_numpy()
    amounts = payments["amount"].to_numpy(dtype=float)
    descriptions = np.array([_canonical_description(d) for d in payments["description"]], dtype=object)
    specific = np.array([_family_specific(d) for d in payments["description"]])
    out = []
    for s, (g, _, _) in enumerate(streams):
        if g is None:
            continue
        members = np.flatnonzero(stream_of == s)
        members = members[np.lexsort((amounts[members], times[members]))]
        family = g
        if g == MUSIC_OR_STREAMING:
            family = _split_music_streaming([hints[i] for i in members], amounts[members], params)
        row = _summarise(client, family, pd.DatetimeIndex(times[members]), amounts[members], refund_counts[s], cutoff, params)
        row["mcc"] = _most_common(mcc[members])
        row["description"] = _most_common(descriptions[members])
        row["family_description_share"] = float(specific[members].mean())
        out.append((row, s))
    out.sort(key=lambda rs: (rs[0]["family"], rs[0]["median_amount"], rs[0]["first_payment"]))
    membership = np.full(len(stream_of), -1)
    for n, (r, s) in enumerate(out):
        r["stream_id"] = n
        membership[stream_of == s] = n
    return [r for r, _ in out], membership, reversed_


def _most_common(values) -> str:
    """The most frequent value; ties go to the smallest, so the result never depends on row order."""
    counts = pd.Series(values, dtype=object).value_counts()
    return str(min(counts[counts == counts.max()].index))


def _match_refunds(
    refunds, stream_of, stream_groups, payment_ns, log_amount, params
) -> tuple[np.ndarray, np.ndarray]:
    """Refunds credited per stream, and which payments they reverse. A refund reverses one
    not-yet-refunded payment that sits in a stream its description allows, precedes it by at most the
    refund window and matches its amount; of several, the closest in amount, then the latest."""
    counts = np.zeros(len(stream_groups), dtype=int)
    refunded = stream_of < 0  # payments outside every stream can never be reversed
    reversed_ = np.zeros(len(stream_of), dtype=bool)
    groups = np.array([stream_groups[s] if s >= 0 else None for s in stream_of], dtype=object)
    window = int(pd.Timedelta(days=params.refund_window_days).value)
    refunds = refunds.sort_values(["timestamp", "amount", "description", "mcc"], kind="stable")
    refund_ns = pd.DatetimeIndex(refunds["timestamp"]).as_unit("ns").asi8
    for t, d, m, a in zip(refund_ns, refunds["description"], refunds["mcc"], refunds["amount"]):
        k, fams, hint = _evidence(d, m)
        if k in ("drop", "stray"):
            continue
        gap = np.abs(log_amount - math.log(max(float(a), 1e-9)))
        reversible = ~refunded & (payment_ns <= t) & (payment_ns >= t - window) & (gap <= params.amount_tolerance)
        for allowed in _group_choices(fams, hint):
            candidates = np.flatnonzero(reversible & np.isin(groups, list(allowed)))
            if len(candidates):
                i = candidates[np.lexsort((-payment_ns[candidates], gap[candidates]))[0]]
                refunded[i] = reversed_[i] = True
                counts[stream_of[i]] += 1
                break
    return counts, reversed_


def _join_on_schedule(stray, stream_of, ranges, days, log_amount, params) -> None:
    """Stray payments (indices `stray`, in time order) join streams in place (`stream_of`). A payment
    joins a stream of at least two payments and a period of at least `stray_min_period_days` when its
    amount fits the stream's amount range (`ranges`, None for a stream merged away) and its date fits
    an empty slot of the stream's schedule: within the schedule tolerance of a whole number of periods
    from one of its payments, at least half a period from all of them and at most one period before
    its first or after its last. Of several streams, the nearest slot wins, then the nearest amount.
    Passes repeat until none joins, so a run of stray payments can extend a stream one period at a
    time. The period floor keeps a stream with a few days' period from chaining up strays."""
    tolerance = params.schedule_tolerance_days
    while True:
        slots = {}
        for s, fitted in enumerate(ranges):
            times = np.sort(days[stream_of == s]) if fitted is not None else np.zeros(0)
            period = _period(np.diff(times), params) if len(times) >= 2 else np.nan
            if period >= params.stray_min_period_days:
                slots[s] = (times, period)
        joined = False
        for i in stray:
            if stream_of[i] >= 0:
                continue
            best = None
            for s, (times, period) in slots.items():
                lo, hi = ranges[s]
                gap = max(lo - log_amount[i], log_amount[i] - hi, 0.0)
                apart = days[i] - times
                if (
                    gap > params.amount_tolerance
                    or np.abs(apart).min() < period / 2
                    or not times[0] - period - tolerance <= days[i] <= times[-1] + period + tolerance
                ):
                    continue
                off = float(np.abs(apart - np.round(apart / period) * period).min())
                if off <= tolerance and (best is None or (off, gap) < best[:2]):
                    best = (off, gap, s)
            if best is not None:
                s = best[2]
                stream_of[i] = s
                slots[s] = (np.sort(np.append(slots[s][0], days[i])), slots[s][1])
                joined = True
        if not joined:
            return


def _split_music_streaming(hints, amounts, params) -> str:
    counts = pd.Series([h for h in hints if h]).value_counts()
    if len(counts) and (len(counts) == 1 or counts.iloc[0] > counts.iloc[1]):
        return counts.index[0]
    return "music" if np.median(amounts) < params.music_streaming_split else "streaming"


def _period(gaps: np.ndarray, params: StreamParams) -> float:
    """Median gap after folding missed payments (a doubled gap counts as two periods)."""
    median = float(np.median(gaps))
    base = min(params.canonical_periods, key=lambda p: abs(p - median))
    folds = np.maximum(1, np.round(gaps / base))
    return float(np.median(gaps / folds))


def _summarise(client, family, times, amounts, n_refunds, cutoff, params) -> dict:
    n = len(times)
    first, last = times[0], times[-1]
    period = gap_mad = next_payment = np.nan
    if n > 1:
        gaps = np.asarray((times[1:] - times[:-1]) / pd.Timedelta(days=1), dtype=float)
        period = _period(gaps, params)
        folded = gaps / np.maximum(1, np.round(gaps / period))
        gap_mad = float(np.median(np.abs(folded - period)))
        step = pd.Timedelta(days=period)
        next_payment = last + step
        if next_payment < cutoff:
            next_payment += step * math.ceil((cutoff - next_payment) / step)
    days_since_last = (cutoff - last) / pd.Timedelta(days=1)
    active = bool(n >= params.min_payments and n > 1 and days_since_last <= params.active_periods * period)
    return {
        "client_id": client,
        "family": family,
        "period_days": period,
        "median_amount": float(np.median(amounts)),
        "amount_cv": float(np.std(amounts, ddof=1) / np.mean(amounts)) if n > 1 else np.nan,
        "gap_mad_days": gap_mad,
        "n_payments": n,
        "first_payment": first,
        "last_payment": last,
        "next_payment": next_payment if n > 1 else pd.NaT,
        "active": active,
        "n_refunds": int(n_refunds),
        "refund_rate": float(n_refunds) / n,
    }


# --- Pseudo-Labels ------------------------------------------------------------


@dataclass(frozen=True)
class LabellerParams:
    """The Pseudo-Label labeller's settings; the defaults label every repeating stream and churn no one."""

    min_payments: int = 2  # payments a Recurring Stream needs in all for its Horizon payment to count
    min_payments_before: int = 0  # of those, payments before the Shifted Cutoff (1: no stream new in the Horizon)
    churn: float = 0.0  # share of Clients whose streams all stop at the Shifted Cutoff: they get `none`


def pseudo_labels(
    transactions: pd.DataFrame,
    cutoff: pd.Timestamp = SHIFTED_CUTOFF,
    horizon: pd.Timedelta = HORIZON,
    min_payments: int = 2,
    params: StreamParams = StreamParams(),
    min_payments_before: int = 0,
    churn: float = 0.0,
) -> pd.Series:
    """Each Client's Pseudo-Label at `cutoff`, indexed by `client_id` (sorted), for every Client in
    `transactions`: the Merchant Family of its first payment in the Horizon (from `cutoff` up to, not
    including, `cutoff + horizon`) that belongs to a Recurring Stream of at least `min_payments`
    payments, `min_payments_before` of them before `cutoff`, or `none`.

    Streams are detected over the Client's whole known history, before and after `cutoff`, so a
    stream that starts inside the Horizon counts once it has repeated (unless `min_payments_before`
    is 1 or more). Decoy Transactions, shop payments and refunds never join a stream; a one-off
    payment forms a stream too short to count; a Filler Description that joins a stream takes that
    stream's family. A Horizon that ends after the last observed day is refused: its Pseudo-Labels
    would come from a partly observed Horizon.

    `churn` is the share of Clients whose streams all stop at `cutoff`: they get `none` whatever they
    paid in the Horizon. Streams inside the known history almost never stop, but at the real Cutoff
    many do, so without churn the Pseudo-Label task has far fewer `none` Clients with live streams.
    Which Clients churn is a fixed draw per Client and Shifted Cutoff (`_churn_draw`), so tables are
    reproducible, a larger share churns a superset of Clients, and a Client pooled at two Shifted
    Cutoffs churns at each independently.
    """
    labeller = LabellerParams(min_payments, min_payments_before, churn)
    return pseudo_label_sweep(transactions, (labeller,), cutoff, horizon, params)[labeller]


def pseudo_label_sweep(
    transactions: pd.DataFrame,
    settings: tuple[LabellerParams, ...],
    cutoff: pd.Timestamp = SHIFTED_CUTOFF,
    horizon: pd.Timedelta = HORIZON,
    params: StreamParams = StreamParams(),
) -> dict[LabellerParams, pd.Series]:
    """`pseudo_labels` for several labeller settings from one stream detection pass."""
    end = cutoff + horizon
    if end > HISTORY_END:
        last_day = (HISTORY_END - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raise ValueError(
            f"the {horizon.days}-day Horizon after Shifted Cutoff {cutoff:%Y-%m-%d} runs past the last "
            f"observed day ({last_day}), so its Pseudo-Labels would come from a partly observed Horizon; "
            f"the latest Shifted Cutoff is {HISTORY_END - horizon:%Y-%m-%d}"
        )
    for s in settings:
        if s.min_payments < 2:
            raise ValueError(f"min_payments must be at least 2 (a Recurring Stream repeats), got {s.min_payments}")
        if s.min_payments_before < 0:
            raise ValueError(f"min_payments_before must be 0 or more, got {s.min_payments_before}")
        if not 0.0 <= s.churn < 1.0:
            raise ValueError(f"churn must be a share of Clients from 0 up to 1, got {s.churn}")

    clients = pd.Index(sorted(transactions["client_id"].astype(str).unique()), dtype="string", name="client_id")
    labels = {s: pd.Series(NONE_LABEL, index=clients, name=LABEL_COLUMN, dtype="string") for s in settings}
    for client, payments, rows, membership, _ in _detect_per_client(transactions, HISTORY_END, params):
        family = np.array([r["family"] for r in rows] + [None], dtype=object)  # [-1] -> no stream
        counts = np.array([r["n_payments"] for r in rows] + [0])
        times = pd.DatetimeIndex(payments["timestamp"])
        in_horizon = (times >= cutoff) & (times < end)
        # each stream's payments before the Shifted Cutoff, and 0 for no stream
        before = np.append(np.bincount(membership[(times < cutoff) & (membership >= 0)], minlength=len(rows)), 0)
        draw = _churn_draw(client, cutoff)
        for s, client_labels in labels.items():
            if draw < s.churn:
                continue  # the Client's streams all stop at the Shifted Cutoff
            long_enough = (counts[membership] >= s.min_payments) & (before[membership] >= s.min_payments_before)
            counted = in_horizon & long_enough
            if counted.any():
                # the earliest counted payment; simultaneous payments of two families go to the first by name
                idx = np.flatnonzero(counted)
                first = idx[np.lexsort((family[membership[idx]].astype(str), times.asi8[idx]))[0]]
                client_labels[client] = family[membership[first]]
    return labels


def _churn_draw(client: str, cutoff: pd.Timestamp) -> float:
    """A fixed draw in [0, 1) for a Client at a Shifted Cutoff, uniform over Clients: the Client churns
    at that Cutoff under any churn share above it."""
    digest = hashlib.sha1(f"{client}@{pd.Timestamp(cutoff).isoformat()}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


# --- per-split cache ----------------------------------------------------------


def cached_streams(
    raw_dir: Path,
    split: str,
    cache_dir: Path,
    cutoff: pd.Timestamp = CUTOFF,
    params: StreamParams = StreamParams(),
) -> tuple[pd.DataFrame, bool]:
    """The split's stream table and whether it came from the cache.

    A table is cached as `<split>-<source key>-<parameter key>.pkl`. The source key covers the raw
    transactions file (size and mtime), the detector code and the family table; the parameter key
    covers the Cutoff and every parameter. So no changed input, code or threshold ever serves a
    stale table. Tables for different parameters coexist, so a sweep does not evict on every
    change; tables built from other sources can never be served again and are evicted.
    """
    source_key = _source_key(raw_dir, split)
    path = Path(cache_dir) / f"{split}-{source_key}-{_digest((str(cutoff), params))}.pkl"
    if path.exists():
        return pd.read_pickle(path), True
    table = detect_streams(data.load_transactions(raw_dir, split), cutoff, params)
    _store(table, path, split, source_key)
    return table, False


def cached_pseudo_labels(
    raw_dir: Path,
    split: str,
    cache_dir: Path,
    settings: tuple[LabellerParams, ...],
    cutoff: pd.Timestamp = SHIFTED_CUTOFF,
    params: StreamParams = StreamParams(),
    horizon: pd.Timedelta = HORIZON,
) -> dict[LabellerParams, tuple[pd.Series, bool]]:
    """The split's Pseudo-Labels at `cutoff` for each labeller setting, and whether each came from the
    cache. Keyed like `cached_streams`, plus the Shifted Cutoff, the Horizon and every labeller
    setting; the settings not yet cached are labelled together in one stream detection pass.
    Reads transactions only, never a label file."""
    source_key = _source_key(raw_dir, split)
    paths = {
        s: Path(cache_dir) / f"{split}-{source_key}-{_digest((str(cutoff), str(horizon), s, params))}.pkl"
        for s in settings
    }
    result = {s: (pd.read_pickle(path), True) for s, path in paths.items() if path.exists()}
    missing = tuple(s for s in paths if s not in result)
    if missing:
        labelled = pseudo_label_sweep(data.load_transactions(raw_dir, split), missing, cutoff, horizon, params)
        for s, labels in labelled.items():
            _store(labels, paths[s], split, source_key)
            result[s] = (labels, False)
    return {s: result[s] for s in settings}


def _source_key(raw_dir: Path, split: str) -> str:
    """The raw transactions file (size and mtime), the detector code and the family table."""
    stat = (Path(raw_dir) / data.TRANSACTION_FILES[split]).stat()
    return _digest((stat.st_size, stat.st_mtime_ns, _detector_version(), _canonical(_family_table())))


def _store(obj, path: Path, split: str, source_key: str) -> None:
    """Pickle `obj` at `path`, evicting the split's entries built from any other source."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob(f"{split}-*.pkl"):
        if not stale.name.startswith(f"{split}-{source_key}-"):
            stale.unlink()
    obj.to_pickle(path)


# The detector's code: this module plus the loaders and constants it reads.
_DETECTOR_SOURCES = ("streams.py", "data.py", "config.py")


def _detector_version() -> str:
    here = Path(__file__).parent
    return hashlib.sha1(b"".join((here / name).read_bytes() for name in _DETECTOR_SOURCES)).hexdigest()


def _family_table() -> tuple:
    """Everything that maps a description and MCC to Merchant Family evidence."""
    return (
        HOME_MCC,
        _ABBREVIATIONS,
        _SHOP_WORDS,
        _DECOY_WORDS,
        _FAMILY_WORDS,
        _MUSIC_HINTS,
        _STREAMING_HINTS,
        _AMBIGUOUS,
    )


def _canonical(obj):
    """A repr-stable form: sets and dict items sorted, since set order varies between interpreters."""
    if isinstance(obj, dict):
        return sorted((repr(k), _canonical(v)) for k, v in obj.items())
    if isinstance(obj, (set, frozenset)):
        return sorted(repr(_canonical(v)) for v in obj)
    if isinstance(obj, (tuple, list)):
        return [_canonical(v) for v in obj]
    return obj


def _digest(obj) -> str:
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
