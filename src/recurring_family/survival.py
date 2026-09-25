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

The soft race (`soft=True`, ticket 14) treats each stream's next payment date as uncertain instead of
fixing the order: T_i = mu_i + sigma_i Z_i with mu_i the stream's race slot (`race_slot`; it needs the
`monthly-slot` order, so every stream has a date) and sigma_i from its own schedule jitter (`jitter_scale`:
max(JITTER_FLOOR_DAYS, JITTER_MAD_SCALE * gap_mad_days), or JITTER_UNPROJECTED_DAYS for a single-payment
stream), calibrated on train history in `experiments/analysis/survival/jitter_calibration.py`. Averaged
over every order the dates can produce, with independent normal dates,

    P(stream i is next) = s_i * INT f_i(t) prod_{j != i} (1 - s_j F_j(t)) dt      (Gauss-Hermite quadrature)
    P(`none`)           = prod_j (1 - s_j)                                       (order-free, unchanged)

The fit is the hard race's (`training_rows`): on train out-of-fold the soft race over the hard fit gained
+0.006 nested tuned macro-F1, while weighting the fit's rows by the same uncertainty (`soft_fit=True`,
`soft_training_rows`: a stream j of another family gets its target-0 row with weight P(T_j < T_m), m being
the label's stream at weight 1; rows lighter than MIN_SOFT_WEIGHT are dropped; the expectation over orders
of the hard race's log-likelihood) lost 0.010. The soft fit stays reachable from Python only.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import ndtr

from .config import CUTOFF, LABELS, MERCHANT_FAMILIES, NONE_LABEL
from .ranker import FEATURE_COLUMNS, LGBM_PARAMS, _params_dict, _params_from, _streams_of, candidates
from .streams import StreamParams

ORDERS = ("unprojected-last", "recent-first", "monthly-slot")
DEFAULT_ORDER = "monthly-slot"
# what a model saved before the order was a setting raced by
_SAVED_WITHOUT_ORDER = "unprojected-last"

MONTH_DAYS = 30.4  # the detector's monthly period

# the soft race's jitter model (ticket 14): sigma of a stream's next payment date around its race slot, in
# days, from `jitter_calibration.py` on train plus unlabeled history (27,101 streams; leave-last-out
# residuals of the projection; no labels)
JITTER_FLOOR_DAYS = 3.6  # the residual spread is flat at about 3.6 days until gap_mad_days passes 2.6
JITTER_MAD_SCALE = 1.4  # then it rises about 1.4 x gap_mad_days
JITTER_UNPROJECTED_DAYS = 6.2  # a single-payment stream: the spread of two-payment heads' projections
HERMITE_NODES = 48  # quadrature nodes for the average over orders; exact to about 1e-7 per stream while no
# stream's spread is more than about 4x another's (the jitter model's range on this data), and the family
# mass is scaled to 1 - P(none) regardless
MIN_SOFT_WEIGHT = 1e-3  # a soft training row lighter than this is censored under every plausible order
SOFT_ORDER = "monthly-slot"  # the soft race needs a date for every stream


def _monthly_slot(days_since_last: pd.Series) -> pd.Series:
    """Days from the Cutoff to one month after the last payment, rolled forward past the Cutoff by whole
    months (as `streams._summarise` rolls a projection)."""
    slot = MONTH_DAYS - days_since_last
    behind = slot < 0
    return slot.where(~behind, slot + MONTH_DAYS * np.ceil(-slot / MONTH_DAYS))


def race_slot(rows: pd.DataFrame, order: str = DEFAULT_ORDER) -> pd.Series:
    """The day (from the Cutoff) at which each candidate row races under `order`: its projected payment,
    or, with no projection, its monthly slot under `monthly-slot` and +inf (last) otherwise."""
    if order not in ORDERS:
        raise ValueError(f"unknown race order {order!r}; one of {ORDERS}")
    unprojected = rows["days_to_next"].isna()
    key = rows["days_to_next"].fillna(np.inf)
    if order == "monthly-slot":
        key = key.where(~unprojected, _monthly_slot(rows["days_since_last"]))
    return key


def race_order(rows: pd.DataFrame, order: str = DEFAULT_ORDER) -> pd.DataFrame:
    """Candidate rows (`ranker.candidates`) sorted into race order per Client by `order` (one of
    `ORDERS`; see the module docstring), with `order` 0, 1, ... and, unless `unprojected-last`,
    `next_rank` = `order` + 1."""
    key = race_slot(rows, order)
    unprojected = rows["days_to_next"].isna()
    since = pd.Series(0.0, index=rows.index)
    if order == "recent-first":
        since = rows["days_since_last"].where(unprojected, 0.0)
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


# --- the soft race (ticket 14) ------------------------------------------------------------------


def jitter_scale(rows: pd.DataFrame) -> np.ndarray:
    """sigma (days) of each candidate row's next payment date: max(JITTER_FLOOR_DAYS, JITTER_MAD_SCALE *
    gap_mad_days), or JITTER_UNPROJECTED_DAYS for a stream with no projection."""
    mad = rows["gap_mad_days"].to_numpy(dtype=float)
    sigma = np.maximum(JITTER_FLOOR_DAYS, JITTER_MAD_SCALE * np.nan_to_num(mad, nan=0.0))
    return np.where(rows["days_to_next"].isna().to_numpy(), JITTER_UNPROJECTED_DAYS, sigma)


def _padded(table: pd.DataFrame, *columns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Per-Client padding of row arrays: the row's Client code and position, the (Clients x max streams)
    mask of real cells, and each array laid out that way (padding 0)."""
    client = table["client_id"].astype(str).to_numpy()
    codes, uniques = pd.factorize(client)
    pos = pd.Series(np.ones(len(codes), dtype=int)).groupby(codes).cumsum().to_numpy() - 1
    shape = (len(uniques), int(pos.max()) + 1 if len(pos) else 0)
    mask = np.zeros(shape, dtype=bool)
    mask[codes, pos] = True
    out = []
    for column in columns:
        grid = np.zeros(shape, dtype=float)
        grid[codes, pos] = column
        out.append(grid)
    return codes, pos, mask, out


def _over_streams(a: np.ndarray, axis: int, product: bool) -> np.ndarray:
    """A product or sum along `axis`, accumulated one stream at a time. A blocked reduction groups the
    operands by the padded width, which moves the last bit; this way a padded cell (x1 or +0) is exact, so a
    Client's probabilities are the same alone and in any batch."""
    shape = a.shape[:axis] + a.shape[axis + 1 :]
    out = np.ones(shape) if product else np.zeros(shape)
    for k in range(a.shape[axis]):
        piece = np.take(a, k, axis=axis)
        out = out * piece if product else out + piece
    return out


def _soft_next(s: np.ndarray, mu: np.ndarray, sigma: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Padded (Clients x streams) s, mu, sigma -> the quadrature of each stream's P(next) (unnormalised) and
    each Client's P(`none`) = prod (1 - s), which no order changes. Padded cells give 0 and factor 1."""
    z, w = hermegauss(HERMITE_NODES)
    w = w / w.sum()
    sigma = np.where(mask, sigma, 1.0)
    p = np.zeros(mask.shape)
    none = _over_streams(np.where(mask, 1.0 - s, 1.0), axis=1, product=True)
    if not mask.size:
        return p, none
    eye = np.eye(mask.shape[1], dtype=bool)
    for start in range(0, mask.shape[0], 256):  # chunks of Clients: the (C, N, N, K) tensor stays small
        c = slice(start, start + 256)
        t = mu[c][:, :, None] + sigma[c][:, :, None] * z  # (C, N_i, K): where stream i lands
        before = ndtr((t[:, :, None, :] - mu[c][:, None, :, None]) / sigma[c][:, None, :, None])  # (C, N_i, N_j, K)
        factor = 1.0 - s[c][:, None, :, None] * before
        factor = np.where(eye[None, :, :, None] | ~mask[c][:, None, :, None], 1.0, factor)
        p[c] = np.where(mask[c], s[c] * (_over_streams(factor, axis=2, product=True) * w).sum(axis=-1), 0.0)
    return p, none


def soft_race_proba(
    table: pd.DataFrame, s: np.ndarray, mu: np.ndarray, sigma: np.ndarray, clients: pd.Index
) -> pd.DataFrame:
    """`race_proba` averaged over the orders the uncertain dates T_i ~ N(mu_i, sigma_i) can produce (the
    module docstring). P(`none`) is exactly prod (1 - s), and the family mass, the quadrature's answer,
    is scaled to 1 - P(`none`) (the scaling corrects quadrature error only)."""
    clients = pd.Index(clients, name="client_id")
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma > 0).all()):
        raise ValueError("the soft race needs a finite date and a positive spread for every stream")
    codes, pos, mask, (s_, mu_, sigma_) = _padded(table, s, mu, sigma)
    p, none = _soft_next(s_, mu_, sigma_, mask)
    mass = _over_streams(p, axis=1, product=False)
    p = p * np.where(mass > 0, (1.0 - none) / np.where(mass > 0, mass, 1.0), 1.0)[:, None]
    rows = pd.DataFrame(
        {"client_id": table["client_id"].astype(str).to_numpy(), "family": table["family"].astype(str).to_numpy(), "p": p[codes, pos]}
    )
    families = rows.groupby(["client_id", "family"])["p"].sum().unstack("family")
    out = families.reindex(index=clients.astype(str), columns=list(MERCHANT_FAMILIES)).fillna(0.0).astype(float)
    out.index = clients
    out[NONE_LABEL] = (1.0 - out.sum(axis=1)).clip(lower=0.0)
    out = out[list(LABELS)]
    return out.div(out.sum(axis=1), axis=0)


def soft_training_rows(
    table: pd.DataFrame, labels: pd.Series, mu: np.ndarray, sigma: np.ndarray, present: set[str] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """`training_rows` under uncertain dates: which rows the model trains on, their targets, their sample
    weights and the unexplained count. The label's stream m (the first of its family in race order) keeps
    target 1 at weight 1; every stream of another family gets target 0 at weight P(T_j < T_m), unless that
    is under MIN_SOFT_WEIGHT; other streams of the label's family give no row; a `none` Client's streams
    all get target 0 at weight 1."""
    _, target, unexplained = training_rows(table, labels, present)
    truth = pd.Series(labels.astype(str).to_numpy(), index=labels.index.astype(str))
    client = table["client_id"].astype(str)
    label = client.map(truth)
    is_none = (label == NONE_LABEL).to_numpy()
    hit = target.astype(bool)  # the label's stream: target 1 on a kept row
    explained = pd.Series(hit, index=table.index).groupby(client).transform("any").to_numpy()
    same_family = (table["family"].astype(str) == label).to_numpy()
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    mu_m = pd.Series(np.where(hit, mu, np.nan), index=table.index).groupby(client).transform("max").to_numpy()
    sigma_m = pd.Series(np.where(hit, sigma, np.nan), index=table.index).groupby(client).transform("max").to_numpy()
    with np.errstate(invalid="ignore"):
        before = ndtr((mu_m - mu) / np.sqrt(sigma_m**2 + sigma**2))
    weight = np.where(is_none | hit, 1.0, np.nan_to_num(before, nan=0.0))
    keep = (is_none | (explained & (hit | ~same_family))) & (weight >= MIN_SOFT_WEIGHT)
    return keep, hit.astype(int), weight, unexplained


class SurvivalModel:
    """The Survival Race; see the module docstring."""

    name = "survival"

    def __init__(
        self,
        booster: lgb.Booster | None = None,
        constant: float | None = None,
        order: str = DEFAULT_ORDER,
        soft: bool = False,
        soft_fit: bool = False,
        stream_params: StreamParams | None = None,
    ):
        if order not in ORDERS:
            raise ValueError(f"unknown race order {order!r}; one of {ORDERS}")
        if soft and order != SOFT_ORDER:
            raise ValueError(f"the soft race needs the {SOFT_ORDER!r} order (every stream has a date), not {order!r}")
        if soft_fit and not soft:
            raise ValueError("soft_fit weights the fit's rows for the soft race; it needs soft=True")
        self.order = order  # how a Client's streams race (`ORDERS`)
        self.soft = bool(soft)  # average the race over uncertain payment dates (the module docstring)
        self.soft_fit = bool(soft_fit)  # also weight the fit's rows by that uncertainty (lost on train; off)
        self.stream_params = stream_params  # the stream detector's settings; None is the default detector
        self.booster = booster
        self.constant = constant  # every stream's s when training had one outcome only (or no rows)
        self.n_training_rows: int | None = None
        self.n_unexplained: int | None = None  # labelled Clients the race cannot explain, left out of the fit
        self.training_weight: float | None = None  # the soft fit's total sample weight

    def fit(self, transactions: pd.DataFrame, labels: pd.Series) -> "SurvivalModel":
        unknown = set(labels) - set(LABELS)
        if unknown:
            raise ValueError(f"labels outside the allowed set: {sorted(unknown)}")
        table = race_table(_streams_of(transactions, labels.index, self.stream_params), order=self.order)
        present = set(transactions["client_id"].astype(str))
        keep, target, weight, self.n_unexplained = self._training_rows(table, labels, present)
        x, y = table.loc[keep, FEATURE_COLUMNS], target[keep]
        self.n_training_rows = int(keep.sum())
        self.training_weight = float(weight[keep].sum()) if self.soft_fit else None
        self.booster, self.constant = None, None
        if len(set(y)) < 2:
            self.constant = float(np.average(y, weights=weight[keep] if self.soft_fit else None)) if len(y) else 0.0
            return self
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        if self.soft_fit:
            model.fit(x, y, sample_weight=weight[keep])
        else:
            model.fit(x, y)
        self.booster = model.booster_
        return self

    def _training_rows(
        self, table: pd.DataFrame, labels: pd.Series, present: set[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """The fit's rows, targets, weights and unexplained count: `training_rows` at weight 1, or
        `soft_training_rows` under `soft_fit`."""
        if self.soft_fit:
            return soft_training_rows(table, labels, race_slot(table, self.order), jitter_scale(table), present)
        keep, target, unexplained = training_rows(table, labels, present)
        return keep, target, np.ones(len(table)), unexplained

    @property
    def variant(self) -> str:
        """What the experiment log adds to the model name: the race order, unless `unprojected-last`
        (the order of the runs logged before the order was a setting)."""
        order = "" if self.order == "unprojected-last" else f"+{self.order}"
        return order + ("+soft-fit" if self.soft_fit else "+soft" if self.soft else "")

    def summary(self) -> str:
        """One line for `train`: the race order, the fit's rows and the Clients it left out."""
        race = f"soft survival race ({self.order}, uncertain dates)" if self.soft else f"survival race ({self.order})"
        rows = f"{self.n_training_rows} training streams"
        if self.soft_fit and self.training_weight is not None:
            rows += f" (soft fit, total weight {self.training_weight:.1f})"
        return f"{race}: {rows}; {self.n_unexplained} Clients whose label family has no Candidate Stream left out"

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
        table = race_table(_streams_of(transactions, clients, self.stream_params), order=self.order)
        if self.soft:
            return soft_race_proba(table, self.survival(table), race_slot(table, self.order), jitter_scale(table), clients)
        return race_proba(table, self.survival(table), clients)

    def save(self, path: Path) -> None:
        saved = {
            "model": self.name,
            "features": FEATURE_COLUMNS,
            "constant": self.constant,
            "order": self.order,
            "soft": self.soft,
            "soft_fit": self.soft_fit,
            "training_weight": self.training_weight,
            "n_training_rows": self.n_training_rows,
            "n_unexplained": self.n_unexplained,
            "booster": None if self.booster is None else self.booster.model_to_string(),
        }
        if self.stream_params is not None:
            saved["stream_params"] = _params_dict(self.stream_params)
        Path(path).write_text(json.dumps(saved))

    @classmethod
    def load(cls, path: Path) -> "SurvivalModel":
        saved = json.loads(Path(path).read_text())
        if saved.get("features") != FEATURE_COLUMNS:
            raise ValueError(f"{path} was saved with other features; retrain it")
        booster = None if saved["booster"] is None else lgb.Booster(model_str=saved["booster"])
        model = cls(
            booster, saved["constant"], saved.get("order", _SAVED_WITHOUT_ORDER), saved.get("soft", False), saved.get("soft_fit", False)
        )
        model.n_training_rows, model.n_unexplained = saved.get("n_training_rows"), saved.get("n_unexplained")
        model.training_weight = saved.get("training_weight")
        model.stream_params = _params_from(saved.get("stream_params"))
        return model
