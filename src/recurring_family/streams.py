"""Stream detection: Client transactions + Cutoff -> one row per Recurring Stream.

Only outgoing card payments before the Cutoff are candidates. Each candidate gets
Merchant Family evidence from its description and MCC:

- a *family keyword* ("gym", "saas", "audio", ...) names one family. On that family's home MCC it
  starts a stream; on any other MCC it only joins an existing stream of the family, like a filler;
- an *ambiguous* description ("premium plan", "digital plus", "monthly plan") names a set
  of families, and the MCCs of its amount cluster pick one;
- a *Filler Description* ("member plan", "subscription charge", ...) names none. It never
  starts a stream but joins one whose amount and MCC it fits.

Shop descriptions, service fees and Decoy Transactions ("digital order", "merchant charge",
...) are dropped. Music and streaming share MCC 5812: a description hint ("audio", "video", ...)
puts a payment in its family before amount clustering, so two streams at nearby amounts never
chain into one. Unhinted payments ("premium plan") join the hinted stream whose amount they fit;
the rest form streams that are split into music or streaming by amount.

A refund is credited to a stream only when it reverses one of the stream's payments: a payment
of matching amount made shortly before it. Each payment is reversed at most once.
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
from .config import CUTOFF

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
]


@dataclass(frozen=True)
class StreamParams:
    """Every detector threshold, in one place so they can be swept."""

    amount_tolerance: float = 0.06  # log-amount gap that splits two amount clusters
    active_periods: float = 1.6  # Active Stream: last payment within this many periods of the Cutoff
    min_payments: int = 3  # Active Stream: at least this many payments
    canonical_periods: tuple[float, ...] = (14.0, 30.4)  # biweekly and monthly, in days
    music_streaming_split: float = 15.5  # 5812 streams without a hint: below is music, above streaming
    refund_window_days: float = 7.0  # a refund reverses a payment made at most this many days before it


# --- per-row evidence ---------------------------------------------------------


def _words(description: str) -> list[str]:
    return [_ABBREVIATIONS.get(w, w) for w in re.findall(r"[a-z0-9]+", str(description).lower())]


def _evidence(description: str, mcc: str) -> tuple[str, frozenset[str], str | None]:
    """(kind, families, hint): kind is 'family', 'ambiguous', 'filler' or 'drop'."""
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
    return ("filler", frozenset({home}), None) if home else ("drop", frozenset(), None)


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
    tx = transactions[transactions["timestamp"] < cutoff]
    payments = tx[(tx["type"] == "card_payment") & (tx["direction"] == "out")]
    refunds = tx[(tx["type"] == "refund") & (tx["direction"] == "in")]

    rows = []
    for client, group in payments.groupby("client_id", sort=True):
        client_refunds = refunds[refunds["client_id"] == client]
        rows.extend(_client_streams(str(client), group, client_refunds, cutoff, params))
    table = pd.DataFrame(rows, columns=COLUMNS)
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
        }
    )


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

    ns = pd.DatetimeIndex(payments["timestamp"]).as_unit("ns").asi8
    refund_counts = _match_refunds(refunds, stream_of, [g for g, _, _ in streams], ns, log_amount, params)

    times = payments["timestamp"].to_numpy()
    amounts = payments["amount"].to_numpy(dtype=float)
    out = []
    for s, (g, _, _) in enumerate(streams):
        if g is None:
            continue
        members = np.flatnonzero(stream_of == s)
        members = members[np.lexsort((amounts[members], times[members]))]
        family = g
        if g == MUSIC_OR_STREAMING:
            family = _split_music_streaming([hints[i] for i in members], amounts[members], params)
        out.append(_summarise(client, family, pd.DatetimeIndex(times[members]), amounts[members], refund_counts[s], cutoff, params))
    out.sort(key=lambda r: (r["family"], r["median_amount"], r["first_payment"]))
    for n, r in enumerate(out):
        r["stream_id"] = n
    return out


def _match_refunds(refunds, stream_of, stream_groups, payment_ns, log_amount, params) -> np.ndarray:
    """Refunds credited per stream. A refund reverses one not-yet-refunded payment that sits in a stream
    its description allows, precedes it by at most the refund window and matches its amount; of
    several, the closest in amount, then the latest."""
    counts = np.zeros(len(stream_groups), dtype=int)
    refunded = stream_of < 0  # payments outside every stream can never be reversed
    groups = np.array([stream_groups[s] if s >= 0 else None for s in stream_of], dtype=object)
    window = int(pd.Timedelta(days=params.refund_window_days).value)
    refunds = refunds.sort_values(["timestamp", "amount", "description", "mcc"], kind="stable")
    refund_ns = pd.DatetimeIndex(refunds["timestamp"]).as_unit("ns").asi8
    for t, d, m, a in zip(refund_ns, refunds["description"], refunds["mcc"], refunds["amount"]):
        k, fams, hint = _evidence(d, m)
        if k == "drop":
            continue
        gap = np.abs(log_amount - math.log(max(float(a), 1e-9)))
        reversible = ~refunded & (payment_ns <= t) & (payment_ns >= t - window) & (gap <= params.amount_tolerance)
        for allowed in _group_choices(fams, hint):
            candidates = np.flatnonzero(reversible & np.isin(groups, list(allowed)))
            if len(candidates):
                i = candidates[np.lexsort((-payment_ns[candidates], gap[candidates]))[0]]
                refunded[i] = True
                counts[stream_of[i]] += 1
                break
    return counts


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
    source = Path(raw_dir) / data.TRANSACTION_FILES[split]
    stat = source.stat()
    source_key = _digest((stat.st_size, stat.st_mtime_ns, _detector_version(), _canonical(_family_table())))
    path = Path(cache_dir) / f"{split}-{source_key}-{_digest((str(cutoff), params))}.pkl"
    if path.exists():
        return pd.read_pickle(path), True
    table = detect_streams(data.load_transactions(raw_dir, split), cutoff, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob(f"{split}-*.pkl"):
        if not stale.name.startswith(f"{split}-{source_key}-"):
            stale.unlink()
    table.to_pickle(path)
    return table, False


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
