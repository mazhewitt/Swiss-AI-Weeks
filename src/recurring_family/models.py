"""The model interface and the prior-probability model."""

import json
from pathlib import Path
from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import LABELS
from .blend import BlendModel
from .features import DescriptionRates, client_features, out_of_fold_description_rates
from .ranker import RankerModel
from .rules import RulesModel
from .streams import detect_streams
from .survival import SurvivalModel


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


class LgbmModel:
    """E2: LightGBM multiclass on one stream-feature row per Client, with balanced class weights and
    default hyperparameters. Streams are detected from the transactions it is given (ADR 0001)."""

    name = "lgbm"

    def __init__(self, booster=None, classes=None, rates: DescriptionRates | None = None):
        self.booster = booster
        self.classes = classes
        self.rates = rates

    def fit(self, transactions: pd.DataFrame, labels: pd.Series) -> "LgbmModel":
        unknown = set(labels) - set(LABELS)
        if unknown:
            raise ValueError(f"labels outside the allowed set: {sorted(unknown)}")
        streams = _streams_of(transactions, labels.index)
        # the description rate is label-derived: training rows get it out of fold
        x = _out_of_fold_features(streams, labels)
        self.rates = DescriptionRates.fit(streams, labels)
        self.classes = sorted(set(labels))
        self.booster = None
        if len(self.classes) > 1:
            model = lgb.LGBMClassifier(class_weight="balanced", verbose=-1)
            model.fit(x, labels.to_numpy(dtype=object))
            self.classes = [str(c) for c in model.classes_]
            self.booster = model.booster_
        return self

    @staticmethod
    def training_features(transactions: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """The rows `fit` trains on, one per labelled Client. The description rate is label-derived,
        so each row gets it out of fold: no Client's own label reaches its row."""
        return _out_of_fold_features(_streams_of(transactions, labels.index), labels)

    def features(self, transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
        """The feature rows this model scores for Clients it was not fitted on, one per Client in
        `clients`. For its training Clients use `training_features`."""
        if self.rates is None:
            raise RuntimeError("LgbmModel is not fitted")
        streams = _streams_of(transactions, clients)
        return client_features(streams, clients, self.rates.encode(streams))

    def predict_proba(self, transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
        x = self.features(transactions, clients)
        out = pd.DataFrame(0.0, index=x.index, columns=list(LABELS))
        if self.booster is None:  # a single training class
            out[self.classes[0]] = 1.0
            return out
        raw = np.asarray(self.booster.predict(x), dtype=float)
        if raw.ndim == 1:  # two classes: LightGBM fits a binary model and returns P(second class)
            raw = np.column_stack([1.0 - raw, raw])
        out[self.classes] = raw
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "model": self.name,
                    "classes": self.classes,
                    "description_rates": self.rates.to_dict(),
                    "booster": None if self.booster is None else self.booster.model_to_string(),
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "LgbmModel":
        saved = json.loads(Path(path).read_text())
        booster = None if saved["booster"] is None else lgb.Booster(model_str=saved["booster"])
        return cls(booster, saved["classes"], DescriptionRates.from_dict(saved["description_rates"]))


def _streams_of(transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
    return detect_streams(transactions[transactions["client_id"].isin(set(clients))])


def _out_of_fold_features(streams: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    return client_features(streams, labels.index, out_of_fold_description_rates(streams, labels))


MODELS: dict[str, type] = {
    "prior": PriorModel, "rules": RulesModel, "lgbm": LgbmModel, "ranker": RankerModel, "blend": BlendModel,
    "survival": SurvivalModel,
}


def predict_labels(proba: pd.DataFrame) -> pd.Series:
    """Argmax over the allowed labels (ties go to the earlier label in `LABELS`)."""
    return proba[list(LABELS)].idxmax(axis=1)
