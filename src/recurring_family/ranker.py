"""The Stream Ranker: which of a Client's Recurring Streams carries its Next Recurring Family?

Every detected Recurring Stream is a Candidate Stream, whether or not it is Active. A LightGBM
binary model scores each candidate: how likely is its family the Client's label? The features
come only from the stream table (ADR 0001): the candidate's own fields, its rank among the
Client's streams by projected next payment, and Client-level context. No description statistic
and no label-derived encoding enters a feature.

Per Client, each Merchant Family's score is its best candidate's score, `none` is one minus the
best score overall, and the row is normalised to sum to 1. A Client without candidates is `none`.

Pseudo-Labelled Clients (see `pseudo_examples`) can join the fit as extra candidate rows under a
sample weight. They only ever add training rows: every Client the model is asked to score, and so
every out-of-fold row and the decision layer fitted on them, stays a real-labelled one.

With a `none` model (`none_model.NoneModel`, `--none-model`), P(`none`) of a Client with Candidate
Streams comes from that Client-level model instead, and the families share the rest in proportion to
their best candidates' scores. It is fitted on the real-labelled Clients with Candidate Streams only,
never on Pseudo-Labelled ones: a Shifted Cutoff has almost no churn. Its `RANKER_NONE` feature, the
ranker's own `none`, is cross-fitted for the training Clients, so no Client's label reaches its own row.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from .config import CUTOFF, LABELS, MERCHANT_FAMILIES, NONE_LABEL
from .none_model import RANKER_NONE, NoneModel, client_features, combine
from .streams import StreamParams, detect_stream_payments

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
    "n_jobs": 1,  # small data: more threads only add overhead (30x slower here)
    "verbose": -1,
}

# the sample weight of a Pseudo-Labelled Client's candidate rows; a real-labelled one's weigh 1
PSEUDO_WEIGHT = 0.5

# folds of the cross-fit that gives the `none` model's training Clients the ranker's `none`
NONE_FOLDS = 5

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


def pseudo_examples(streams: pd.DataFrame, labels: pd.Series, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Training rows of Pseudo-Labelled Clients at one Shifted Cutoff: their Candidate Streams, from a
    stream table detected at `cutoff` (so on the transactions before it only), each with `target` 1
    when its family is the Client's Pseudo-Label. Clients without candidates add no rows."""
    streams = streams[streams["client_id"].astype(str).isin(set(labels.index.astype(str)))]
    rows = candidates(streams, cutoff)
    rows["target"] = _targets(rows, labels)
    return rows


def _targets(rows: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    """1 where the candidate's family is its Client's label, else 0."""
    truth = labels.astype(str)
    truth.index = truth.index.astype(str)
    return (
        rows["family"].astype(str).to_numpy(dtype=object) == truth.reindex(rows["client_id"]).to_numpy(dtype=object)
    ).astype(int)


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
    decoy, a truncated history) is detected afresh. Only streams and their member payments are kept,
    never labels. One memo serves one detector setting (`params`; None is the default detector).
    """

    def __init__(self, params: StreamParams | None = None):
        self.params = params
        self.table: pd.DataFrame | None = None  # the stream table plus each stream's history key
        self.paid: pd.DataFrame | None = None  # the streams' member payments plus their history key
        self.seen: set[str] = set()

    def streams(self, transactions: pd.DataFrame) -> pd.DataFrame:
        return self.detected(transactions)[0]

    def detected(self, transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The stream table and the streams' member payments (`detect_stream_payments`)."""
        keys = _history_keys(transactions)
        missing = keys[~keys.isin(self.seen)]
        if len(missing):
            fresh, paid = self._detect(transactions[transactions["client_id"].astype(str).isin(set(missing.index))])
            fresh["_key"] = fresh["client_id"].astype(str).map(missing).to_numpy()
            paid["_key"] = paid["client_id"].astype(str).map(missing).to_numpy()
            if self.table is None:
                self.table, self.paid = fresh, paid
            else:
                if len(fresh):
                    self.table = pd.concat([self.table, fresh], ignore_index=True)
                if len(paid):
                    self.paid = pd.concat([self.paid, paid], ignore_index=True)
            self.seen.update(missing)
        if self.table is None:  # no Client history seen yet: an empty, typed stream table
            return self._detect(transactions)
        wanted = set(keys)
        found = self.table[self.table["_key"].isin(wanted)]
        paid = self.paid[self.paid["_key"].isin(wanted)]
        return (
            found.drop(columns="_key").sort_values(["client_id", "stream_id"]).reset_index(drop=True),
            paid.drop(columns="_key")
            .sort_values(["client_id", "stream_id", "timestamp", "amount"], kind="stable")
            .reset_index(drop=True),
        )


    def _detect(self, transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.params is None:
            return detect_stream_payments(transactions)
        return detect_stream_payments(transactions, params=self.params)


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
_MEMOS: dict[StreamParams, _StreamMemo] = {}  # one per non-default detector setting (`stream_params`)


def _memo(params: StreamParams | None) -> _StreamMemo:
    if params is None:
        return _MEMO
    if params not in _MEMOS:
        _MEMOS[params] = _StreamMemo(params)
    return _MEMOS[params]


def _streams_of(transactions: pd.DataFrame, clients: pd.Index, params: StreamParams | None = None) -> pd.DataFrame:
    return _memo(params).streams(transactions[transactions["client_id"].astype(str).isin(set(map(str, clients)))])


def _detected(
    transactions: pd.DataFrame, clients: pd.Index, params: StreamParams | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _memo(params).detected(transactions[transactions["client_id"].astype(str).isin(set(map(str, clients)))])


def _params_dict(params: StreamParams | None) -> dict | None:
    """A detector setting as saved with a model (None: the default detector)."""
    return None if params is None else {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(params).items()}


def _params_from(saved: dict | None) -> StreamParams | None:
    if saved is None:
        return None
    return StreamParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()})


# --- the model ------------------------------------------------------------------------------


class RankerModel:
    """Scores every Candidate Stream with a LightGBM binary model; see the module docstring."""

    name = "ranker"

    def __init__(
        self,
        booster: lgb.Booster | None = None,
        constant: float | None = None,
        pseudo: pd.DataFrame | None = None,
        pseudo_weight: float = PSEUDO_WEIGHT,
        none_model: NoneModel | None = None,
        stream_params: StreamParams | None = None,
    ):
        self.booster = booster
        self.constant = constant  # the score of every candidate when training had one outcome only
        # Pseudo-Labelled training rows (`pseudo_examples`), pooled into every fit; never saved
        self.pseudo = pseudo
        self.pseudo_weight = pseudo_weight
        self.none_model = none_model  # the Client-level `none` model, fitted with the ranker when given
        # the stream detector's settings (e.g. `StreamParams(robust=True)`); None is the default detector
        self.stream_params = stream_params

    @property
    def variant(self) -> str:
        """What the experiment log adds to the model name."""
        return "" if self.none_model is None else "+none"

    def _streams(self, transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
        if self.stream_params is None:
            return _streams_of(transactions, clients)
        return _streams_of(transactions, clients, self.stream_params)

    def _detected(self, transactions: pd.DataFrame, clients: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.stream_params is None:
            return _detected(transactions, clients)
        return _detected(transactions, clients, self.stream_params)

    def fit(self, transactions: pd.DataFrame, labels: pd.Series) -> "RankerModel":
        unknown = set(labels) - set(LABELS)
        if unknown:
            raise ValueError(f"labels outside the allowed set: {sorted(unknown)}")
        self._fit_ranker(transactions, labels)
        if self.none_model is not None:
            x = self.none_training_features(transactions, labels)
            truth = pd.Series(labels.astype(str).to_numpy(), index=labels.index.astype(str))
            self.none_model.fit(x, truth.reindex(x.index) == NONE_LABEL)
        return self

    def _fit_ranker(self, transactions: pd.DataFrame, labels: pd.Series) -> None:
        rows = candidates(self._streams(transactions, labels.index))
        x, target, weight = rows[FEATURE_COLUMNS], _targets(rows, labels), None
        if self.pseudo is not None and len(self.pseudo):
            x = pd.concat([x, self.pseudo[FEATURE_COLUMNS]], ignore_index=True)
            target = np.concatenate([target, self.pseudo["target"].to_numpy(dtype=int)])
            weight = np.concatenate([np.ones(len(rows)), np.full(len(self.pseudo), float(self.pseudo_weight))])
        self.booster, self.constant = None, None
        if len(set(target)) < 2:
            self.constant = float(np.average(target, weights=weight)) if len(target) else 0.0
            return
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(x, target, sample_weight=weight)
        self.booster = model.booster_

    def none_training_features(self, transactions: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """The rows the `none` model trains on: one per labelled Client with at least one Candidate
        Stream. `RANKER_NONE` is cross-fitted: each Client's comes from rankers fitted on the other
        folds (with every Pseudo-Labelled row), so no Client's own label reaches its row."""
        streams, payments = self._detected(transactions, labels.index)
        x = client_features(streams, payments)
        if RANKER_NONE in self.none_model.features:
            x.insert(0, RANKER_NONE, self._cross_fitted_none(transactions, labels).reindex(x.index))
        return x

    def _cross_fitted_none(self, transactions: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """Each labelled Client's ranker `none` from a ranker fitted on the other folds. The folds are
        drawn over the Clients, not their labels, so a Client's fold never depends on its own label."""
        clients = pd.Index(labels.index.astype(str), name="client_id")
        out = pd.Series(np.nan, index=clients, name=RANKER_NONE)
        if len(clients) < 2:
            return out
        splitter = KFold(n_splits=min(NONE_FOLDS, len(clients)), shuffle=True, random_state=0)
        for fit_idx, score_idx in splitter.split(clients):
            ranker = RankerModel(pseudo=self.pseudo, pseudo_weight=self.pseudo_weight, stream_params=self.stream_params)
            ranker.fit(transactions, labels.iloc[fit_idx])
            out.iloc[score_idx] = ranker.predict_proba(transactions, labels.index[score_idx])[NONE_LABEL].to_numpy()
        return out

    def predict_proba(self, transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
        if self.booster is None and self.constant is None:
            raise RuntimeError("RankerModel is not fitted")
        rows = candidates(self._streams(transactions, clients))
        if self.booster is None:
            score = np.full(len(rows), self.constant, dtype=float)
        else:
            score = np.asarray(self.booster.predict(rows[FEATURE_COLUMNS]), dtype=float) if len(rows) else np.zeros(0)
        scored = pd.DataFrame({"client_id": rows["client_id"], "family": rows["family"].astype(str), "score": score})
        proba = client_proba(scored, clients)
        if self.none_model is None:
            return proba
        # the `none` model scores every requested Client with a Candidate Stream
        ids = proba.index.astype(str)
        x = client_features(*self._detected(transactions, clients))
        if RANKER_NONE in self.none_model.features:
            x.insert(0, RANKER_NONE, pd.Series(proba[NONE_LABEL].to_numpy(), index=ids).reindex(x.index))
        p_none = self.none_model.predict(x).reindex(ids)
        return combine(proba, pd.Series(p_none.to_numpy(), index=proba.index))

    def save(self, path: Path) -> None:
        saved = {
            "model": self.name,
            "features": FEATURE_COLUMNS,
            "constant": self.constant,
            "booster": None if self.booster is None else self.booster.model_to_string(),
        }
        if self.none_model is not None:
            saved["none_model"] = self.none_model.to_dict()
        if self.stream_params is not None:
            saved["stream_params"] = _params_dict(self.stream_params)
        Path(path).write_text(json.dumps(saved))

    @classmethod
    def load(cls, path: Path) -> "RankerModel":
        saved = json.loads(Path(path).read_text())
        if saved.get("features") != FEATURE_COLUMNS:
            raise ValueError(f"{path} was saved with other ranker features; retrain it")
        booster = None if saved["booster"] is None else lgb.Booster(model_str=saved["booster"])
        none_model = NoneModel.from_dict(saved["none_model"]) if "none_model" in saved else None
        return cls(booster, saved["constant"], none_model=none_model, stream_params=_params_from(saved.get("stream_params")))
