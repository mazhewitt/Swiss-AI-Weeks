"""E1, the rule baseline: each Client's Merchant Family from its surviving Recurring Streams.

A stream *survives* when it is an Active Stream whose projected next payment falls within the
Horizon. An ordering rule picks one surviving stream per Client (default: the one due soonest)
and predicts its Merchant Family; a Client with no surviving stream gets `none`. Nothing is
learned from the labels.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CUTOFF, HORIZON, LABELS, NONE_LABEL
from .streams import StreamParams, detect_streams

# ordering name -> (stream column, ascending): the first stream in this order is picked
ORDERINGS = {
    "soonest": ("next_payment", True),  # projected next payment
    "most_recent": ("last_payment", False),
    "longest": ("n_payments", False),
}
DEFAULT_ORDERING = "soonest"


class RulesModel:
    name = "rules"

    def __init__(self, ordering: str = DEFAULT_ORDERING, params: StreamParams = StreamParams()):
        if ordering not in ORDERINGS:
            raise ValueError(f"unknown ordering {ordering!r}; expected one of {', '.join(ORDERINGS)}")
        self.ordering = ordering
        self.params = params

    def fit(self, transactions: pd.DataFrame, labels: pd.Series) -> "RulesModel":
        unknown = set(labels) - set(LABELS)
        if unknown:
            raise ValueError(f"labels outside the allowed set: {sorted(unknown)}")
        return self

    def surviving_streams(self, transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
        """The Clients' Active Streams due within the Horizon, each Client's in ordering-rule order."""
        streams = detect_streams(transactions[transactions["client_id"].isin(set(clients))], CUTOFF, self.params)
        streams = streams[streams["active"] & (streams["next_payment"] <= CUTOFF + HORIZON)]
        column, ascending = ORDERINGS[self.ordering]
        return streams.sort_values(
            ["client_id", column, "family", "stream_id"], ascending=[True, ascending, True, True], kind="stable"
        )

    def predict_proba(self, transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
        index = pd.Index(clients, name="client_id")
        picked = self.surviving_streams(transactions, index).groupby("client_id")["family"].first()
        predicted = picked.reindex(index).astype(object).fillna(NONE_LABEL)
        onehot = (predicted.to_numpy()[:, None] == np.array(LABELS, dtype=object)[None, :]).astype(float)
        return pd.DataFrame(onehot, index=index, columns=list(LABELS))

    def diagnostics(self, transactions: pd.DataFrame, truth: pd.Series, predicted: pd.Series) -> dict[str, float]:
        """Over the Clients whose label is a Merchant Family: coverage is the share whose true family is
        among their surviving streams; selection accuracy is the share of those covered Clients for which
        the rule picked the true family. NaN when there is nothing to divide by."""
        surviving = self.surviving_streams(transactions, truth.index)
        families = surviving.groupby("client_id")["family"].agg(set)
        with_family = truth[truth != NONE_LABEL]
        covered = pd.Series(
            [label in families.get(client, set()) for client, label in with_family.items()],
            index=with_family.index, dtype=bool,
        )
        correct = predicted.reindex(with_family.index) == with_family
        return {
            "coverage": float(covered.mean()) if len(covered) else float("nan"),
            "selection_accuracy": float(correct[covered].mean()) if covered.any() else float("nan"),
        }

    def save(self, path: Path) -> None:
        payload = {"model": self.name, "ordering": self.ordering, "params": dataclasses.asdict(self.params)}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "RulesModel":
        payload = json.loads(Path(path).read_text())
        params = {k: tuple(v) if isinstance(v, list) else v for k, v in payload["params"].items()}
        return cls(payload["ordering"], StreamParams(**params))
