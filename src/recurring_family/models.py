"""The model interface and the prior-probability model."""

import json
from pathlib import Path
from typing import Protocol

import pandas as pd

from .config import LABELS
from .rules import RulesModel


class Model(Protocol):
    """Fitted on labelled Clients; returns one probability per allowed label per Client.

    `fit` takes the Clients' transactions and a Series of labels indexed by client_id.
    `predict_proba` returns a DataFrame indexed by client_id (in the order given)
    with exactly the columns `LABELS`, each row summing to 1.
    """

    name: str

    def fit(self, transactions: pd.DataFrame, labels: pd.Series) -> "Model": ...

    def predict_proba(self, transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "Model": ...


class PriorModel:
    """Every Client gets the class frequencies of the training labels."""

    name = "prior"

    def __init__(self, priors: dict[str, float] | None = None):
        self.priors = priors

    def fit(self, transactions: pd.DataFrame, labels: pd.Series) -> "PriorModel":
        unknown = set(labels) - set(LABELS)
        if unknown:
            raise ValueError(f"labels outside the allowed set: {sorted(unknown)}")
        freq = labels.value_counts(normalize=True)
        self.priors = {label: float(freq.get(label, 0.0)) for label in LABELS}
        return self

    def predict_proba(self, transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
        if self.priors is None:
            raise RuntimeError("PriorModel is not fitted")
        row = [self.priors[label] for label in LABELS]
        index = pd.Index(clients, name="client_id")
        return pd.DataFrame([row] * len(index), index=index, columns=list(LABELS))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"model": self.name, "priors": self.priors}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "PriorModel":
        return cls(json.loads(Path(path).read_text())["priors"])


MODELS: dict[str, type] = {"prior": PriorModel, "rules": RulesModel}


def predict_labels(proba: pd.DataFrame) -> pd.Series:
    """Argmax over the allowed labels (ties go to the earlier label in `LABELS`)."""
    return proba[list(LABELS)].idxmax(axis=1)
