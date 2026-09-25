"""The Survival Race: each Candidate Stream survives the Cutoff or not, and the soonest survivor wins.

A LightGBM binary model gives every Candidate Stream a survival probability s, from exactly the Stream
Ranker's features (`ranker.candidates`, so the stream table only: ADR 0001). A Client's streams race in
projected payment order, by one of `ORDERS` (`DEFAULT_ORDER` unless the model is given another):

- `unprojected-last`: `days_to_next` ascending; streams with no projection (one payment) last; ties to the
  stream with more payments, then stream-table order. The ranker's own order: ties among unprojected
  streams fall back to stream-table order, which is alphabetical by family.
- `recent-first`: the same, except that within the unprojected block the most recently paid stream
  (`days_since_last` ascending) goes first, then more payments, then stream-table order.
- `monthly-slot`: an unprojected stream races at a slot one month (30.4 days) after its payment, rolled
  forward past the Cutoff as the detector rolls a projection. The slot orders only: its `days_to_next`
  feature stays missing. Ties as in `unprojected-last`.

Under the last two, `next_rank` is the stream's place in the race (`order` + 1), so the feature matches
the race; under `unprojected-last` it already is. Then

    P(stream i is next) = s_i * prod_{j before i} (1 - s_j)
    P(`none`)           = prod_j (1 - s_j)

and a Merchant Family's probability is the sum over its streams. A Client without candidates is `none`.

The fit is a plain binary log-loss over a filtered row set, which is the race's likelihood. When the
label is a family, its first stream in race order takes target 1 and every stream before it target 0;
the streams after it are censored (it is not known whether they would have paid) and give no row. A
`none` Client gives every stream target 0. A Client whose label family has no Candidate Stream is not
explained by the race and gives no rows; `n_unexplained` counts them (among the Clients whose
transactions the fit was given).

No Pseudo-Labels and no `none` model: a Shifted Cutoff has almost no churn, so it would teach s = 1.
Streams come from the ranker's detector path, so both models share one stream cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CUTOFF, LABELS, MERCHANT_FAMILIES, NONE_LABEL
from .ranker import FEATURE_COLUMNS, LGBM_PARAMS, _streams_of, candidates

ORDERS = ("unprojected-last", "recent-first", "monthly-slot")
DEFAULT_ORDER = "monthly-slot"
# what a model saved before the order was a setting raced by
_SAVED_WITHOUT_ORDER = "unprojected-last"

MONTH_DAYS = 30.4  # the detector's monthly period


def _monthly_slot(days_since_last: pd.Series) -> pd.Series:
    """Days from the Cutoff to one month after the last payment, rolled forward past the Cutoff by whole
    months (as `streams._summarise` rolls a projection)."""
    slot = MONTH_DAYS - days_since_last
    behind = slot < 0
    return slot.where(~behind, slot + MONTH_DAYS * np.ceil(-slot / MONTH_DAYS))


def race_order(rows: pd.DataFrame, order: str = DEFAULT_ORDER) -> pd.DataFrame:
    """Candidate rows (`ranker.candidates`) sorted into race order per Client by `order` (one of
    `ORDERS`; see the module docstring), with `order` 0, 1, ... and, unless `unprojected-last`,
    `next_rank` = `order` + 1."""
    if order not in ORDERS:
        raise ValueError(f"unknown race order {order!r}; one of {ORDERS}")
    unprojected = rows["days_to_next"].isna()
    key = rows["days_to_next"].fillna(np.inf)
    since = pd.Series(0.0, index=rows.index)
    if order == "recent-first":
        since = rows["days_since_last"].where(unprojected, 0.0)
    elif order == "monthly-slot":
        key = key.where(~unprojected, _monthly_slot(rows["days_since_last"]))
    keyed = rows.assign(_key=key, _since=since, _n=-rows["n_payments"])
    out = keyed.sort_values(["client_id", "_key", "_since", "_n"], kind="stable").drop(columns=["_key", "_since", "_n"])
    out["order"] = out.groupby("client_id").cumcount()
    if order != "unprojected-last":
        out["next_rank"] = (out["order"] + 1).astype(float)
    return out.reset_index(drop=True)


def race_table(streams: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF, order: str = DEFAULT_ORDER) -> pd.DataFrame:
    """The Candidate Streams of a stream table, in race order: `client_id`, `FEATURE_COLUMNS`, `order`."""
    return race_order(candidates(streams, cutoff), order)


def training_rows(
    table: pd.DataFrame, labels: pd.Series, present: set[str] | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Which rows of a race-ordered table (`race_order`) the model trains on, their targets, and how many
    labelled Clients the race cannot explain (label a family with no Candidate Stream of it). Given
    `present`, only those Clients count as unexplained: the ones whose transactions were given."""
    truth = pd.Series(labels.astype(str).to_numpy(), index=labels.index.astype(str))
    if present is not None:
        present = set(map(str, present))
    client = table["client_id"].astype(str)
    label = client.map(truth)
    hit = table["family"].astype(str).to_numpy(dtype=object) == label.to_numpy(dtype=object)
    hit = pd.Series(hit, index=table.index)
    first_hit = hit & (hit.astype(int).groupby(client).cumsum() == 1)
    stop = table["order"].where(first_hit).groupby(client).transform("min")
    is_none = label == NONE_LABEL
    keep = (is_none | stop.notna()) & (is_none | (table["order"] <= stop))
    target = (first_hit & keep).astype(int)
    explained = set(client[first_hit])
    unexplained = int(
        sum(1 for c, l in truth.items() if l != NONE_LABEL and c not in explained and (present is None or c in present))
    )
    return keep.to_numpy(), target.to_numpy(), unexplained


def race_proba(table: pd.DataFrame, s: np.ndarray, clients: pd.Index) -> pd.DataFrame:
    """Survival probabilities of a race-ordered table's rows -> one row per Client in `clients` with
    exactly `LABELS`, summing to 1. A Client without rows is certainly `none`."""
    clients = pd.Index(clients, name="client_id")
    client = table["client_id"].astype(str).to_numpy()
    s = pd.Series(np.clip(np.asarray(s, dtype=float), 0.0, 1.0), index=table.index)
    # prod_{j before i} (1 - s_j): the Client's running product, shifted by one stream; a certain survivor
    # (s = 1) leaves every stream behind it exactly 0
    before = (1.0 - s).groupby(client).cumprod().groupby(client).shift(1, fill_value=1.0)
    p = pd.DataFrame({"client_id": client, "family": table["family"].astype(str).to_numpy(), "p": s * before})
    families = p.groupby(["client_id", "family"])["p"].sum().unstack("family")
    out = families.reindex(index=clients.astype(str), columns=list(MERCHANT_FAMILIES)).fillna(0.0).astype(float)
    out.index = clients
    out[NONE_LABEL] = (1.0 - out.sum(axis=1)).clip(lower=0.0)
    out = out[list(LABELS)]
    return out.div(out.sum(axis=1), axis=0)


class SurvivalModel:
    """The Survival Race; see the module docstring."""

    name = "survival"

    def __init__(self, booster: lgb.Booster | None = None, constant: float | None = None, order: str = DEFAULT_ORDER):
        if order not in ORDERS:
            raise ValueError(f"unknown race order {order!r}; one of {ORDERS}")
        self.order = order  # how a Client's streams race (`ORDERS`)
        self.booster = booster
        self.constant = constant  # every stream's s when training had one outcome only (or no rows)
        self.n_training_rows: int | None = None
        self.n_unexplained: int | None = None  # labelled Clients the race cannot explain, left out of the fit

    def fit(self, transactions: pd.DataFrame, labels: pd.Series) -> "SurvivalModel":
        unknown = set(labels) - set(LABELS)
        if unknown:
            raise ValueError(f"labels outside the allowed set: {sorted(unknown)}")
        table = race_table(_streams_of(transactions, labels.index), order=self.order)
        present = set(transactions["client_id"].astype(str))
        keep, target, self.n_unexplained = training_rows(table, labels, present)
        x, y = table.loc[keep, FEATURE_COLUMNS], target[keep]
        self.n_training_rows = int(keep.sum())
        self.booster, self.constant = None, None
        if len(set(y)) < 2:
            self.constant = float(np.mean(y)) if len(y) else 0.0
            return self
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(x, y)
        self.booster = model.booster_
        return self

    def summary(self) -> str:
        """One line for `train`: the fit's rows and the Clients it left out."""
        return (
            f"survival race: {self.n_training_rows} training streams; "
            f"{self.n_unexplained} Clients whose label family has no Candidate Stream left out"
        )

    def survival(self, table: pd.DataFrame) -> np.ndarray:
        """s of every row of a race table."""
        if self.booster is None and self.constant is None:
            raise RuntimeError("SurvivalModel is not fitted")
        if self.booster is None:
            return np.full(len(table), self.constant, dtype=float)
        if not len(table):
            return np.zeros(0)
        return np.asarray(self.booster.predict(table[FEATURE_COLUMNS]), dtype=float)

    def predict_proba(self, transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
        table = race_table(_streams_of(transactions, clients), order=self.order)
        return race_proba(table, self.survival(table), clients)

    def save(self, path: Path) -> None:
        saved = {
            "model": self.name,
            "features": FEATURE_COLUMNS,
            "constant": self.constant,
            "order": self.order,
            "n_training_rows": self.n_training_rows,
            "n_unexplained": self.n_unexplained,
            "booster": None if self.booster is None else self.booster.model_to_string(),
        }
        Path(path).write_text(json.dumps(saved))

    @classmethod
    def load(cls, path: Path) -> "SurvivalModel":
        saved = json.loads(Path(path).read_text())
        if saved.get("features") != FEATURE_COLUMNS:
            raise ValueError(f"{path} was saved with other features; retrain it")
        booster = None if saved["booster"] is None else lgb.Booster(model_str=saved["booster"])
        model = cls(booster, saved["constant"], saved.get("order", _SAVED_WITHOUT_ORDER))
        model.n_training_rows, model.n_unexplained = saved.get("n_training_rows"), saved.get("n_unexplained")
        return model
