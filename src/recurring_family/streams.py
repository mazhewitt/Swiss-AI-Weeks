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
...) are dropped. Music and streaming share MCC 5812, so they are detected together and split
per stream: by description hint first, then by amount.
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
    ({"monthly", "plan"}, frozenset({"mobile"})),
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

    # Anchored payments (a family keyword, or a resolved ambiguous description) form streams.
    stream_of = np.full(len(kind), -1)
    streams = []  # (group, low, high) of log amount
    for g in sorted({x for x in group if x is not None}):
        idx = np.flatnonzero(group == g)
        cid = _amount_clusters(log_amount[idx], params.amount_tolerance)
        for c in np.unique(cid):
            members = idx[cid == c]
            stream_of[members] = len(streams)
            streams.append((g, log_amount[members].min(), log_amount[members].max()))

    def fit(log_a, allowed):
        best, best_d = -1, math.inf
        for s, (g, lo, hi) in enumerate(streams):
            if g not in allowed:
                continue
            d = max(lo - log_a, log_a - hi, 0.0)
            if d <= params.amount_tolerance and d < best_d:
                best, best_d = s, d
        return best

    # Filler and unresolved rows join the stream whose amount and family they fit; else dropped.
    for i in np.flatnonzero(group == None):  # noqa: E711
        if kind[i] in ("filler", "ambiguous"):
            stream_of[i] = fit(log_amount[i], families[i])

    refund_counts = np.zeros(len(streams), dtype=int)
    for d, m, a in zip(refunds["description"], refunds["mcc"], refunds["amount"]):
        k, fams, _ = _evidence(d, m)
        if k != "drop":
            s = fit(math.log(max(float(a), 1e-9)), fams)
            if s >= 0:
                refund_counts[s] += 1

    times = payments["timestamp"].to_numpy()
    amounts = payments["amount"].to_numpy(dtype=float)
    out = []
    for s, (g, _, _) in enumerate(streams):
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

    The cache key covers the raw transactions file (size and mtime), the Cutoff and every
    parameter, so a changed input or threshold never serves a stale table.
    """
    source = Path(raw_dir) / data.TRANSACTION_FILES[split]
    stat = source.stat()
    key = hashlib.sha1(repr((stat.st_size, stat.st_mtime_ns, str(cutoff), params)).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"{split}-{key}.pkl"
    if path.exists():
        return pd.read_pickle(path), True
    table = detect_streams(data.load_transactions(raw_dir, split), cutoff, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob(f"{split}-*.pkl"):
        stale.unlink()
    table.to_pickle(path)
    return table, False
