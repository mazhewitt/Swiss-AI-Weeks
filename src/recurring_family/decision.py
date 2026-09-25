"""E3 decision layer: turns any model's probabilities into labels to maximise macro-F1.

A decision is one weight per label plus an optional `none` threshold t:

- without a threshold, each Client gets the label with the highest weighted probability
  (uniform weights reproduce plain argmax exactly, ties included);
- with a threshold, each Client gets its best weighted Merchant Family, unless
  P(none) >= 1 - t, when it gets `none`. So t = 0 never predicts `none` and t = 1
  always does.

`fit` grid-searches the weights and the threshold on out-of-fold probabilities. It
keeps argmax unless a search beats it outright, so on its own fitting data the tuned
decision never scores below argmax.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABELS, MERCHANT_FAMILIES, NONE_LABEL
from .data import DataError
from .evaluation import score

# Each weight is searched over a geometric grid around 1 (quarter octaves, 1/4 .. 4).
WEIGHT_GRID = tuple(float(w) for w in 2.0 ** (np.arange(-8, 9) / 4))
THRESHOLD_GRID = tuple(float(t) for t in np.round(np.arange(0, 101) / 100, 2))
SEARCH_PASSES = 3


@dataclass
class Decision:
    weights: dict[str, float] = field(default_factory=lambda: {label: 1.0 for label in LABELS})
    none_threshold: float | None = None
    # the Clients whose out-of-fold probabilities it was fitted on; never score it on them
    fitted_on_clients: list[str] = field(default_factory=list)

    def apply(self, proba: pd.DataFrame) -> pd.Series:
        weighted = proba[list(LABELS)] * pd.Series(self.weights)[list(LABELS)]
        if self.none_threshold is None:
            return weighted.idxmax(axis=1)
        predicted = weighted[list(MERCHANT_FAMILIES)].idxmax(axis=1)
        t = self.none_threshold
        if t > 0:
            predicted = predicted.where(proba[NONE_LABEL] < 1 - t, NONE_LABEL)
        return predicted

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "weights": self.weights,
                    "none_threshold": self.none_threshold,
                    "fitted_on_clients": list(self.fitted_on_clients),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "Decision":
        raw = json.loads(Path(path).read_text())
        weights = {label: float(w) for label, w in raw["weights"].items()}
        missing = set(LABELS) - set(weights)
        if missing:
            raise DataError(f"decision layer {path} has no weight for {sorted(missing)}")
        threshold = raw.get("none_threshold")
        return cls(
            weights=weights,
            none_threshold=None if threshold is None else float(threshold),
            fitted_on_clients=list(raw.get("fitted_on_clients", [])),
        )


def _macro_f1(decision: Decision, proba: pd.DataFrame, labels: pd.Series) -> float:
    return score(labels, decision.apply(proba))["macro_f1"]


def _ascend(
    start: Decision, coordinates: list[str | None], objective: Callable[[Decision], float]
) -> tuple[Decision, float]:
    """Coordinate-wise grid search: each label's weight (or, for None, the `none` threshold)
    in turn over its grid, keeping only strict gains in `objective`."""
    best, best_f1 = start, objective(start)
    for _ in range(SEARCH_PASSES):
        improved = False
        for coordinate in coordinates:
            for value in THRESHOLD_GRID if coordinate is None else WEIGHT_GRID:
                trial = Decision(weights=dict(best.weights), none_threshold=best.none_threshold)
                if coordinate is None:
                    trial.none_threshold = value
                else:
                    trial.weights[coordinate] = value
                f1 = objective(trial)
                if f1 > best_f1:
                    best, best_f1, improved = trial, f1, True
        if not improved:
            break
    return best, best_f1


def _search(objective: Callable[[Decision], float]) -> tuple[Decision, float, float]:
    """E3's search: weighted argmax over all eight labels, and the best weighted family with a
    `none` threshold. The better one is kept only if it beats argmax outright. Returns the
    decision, argmax's objective and the kept decision's objective."""
    best = Decision()
    argmax_f1 = best_f1 = objective(best)
    searches = [
        (Decision(), list(LABELS)),
        (Decision(none_threshold=0.0), [None, *MERCHANT_FAMILIES]),
    ]
    for start, coordinates in searches:
        found, f1 = _ascend(start, coordinates, objective)
        if f1 > best_f1:
            best, best_f1 = found, f1
    return best, argmax_f1, best_f1


def fit(oof_proba: pd.DataFrame, labels: pd.Series) -> tuple[Decision, dict[str, float]]:
    """Grid-search the weights and the `none` threshold on out-of-fold probabilities to
    maximise macro-F1. Two searches run: weighted argmax over all eight labels, and the
    best weighted family with a `none` threshold. The better one is kept only if it beats
    argmax outright. Returns the decision and the argmax and tuned macro-F1 on this data."""
    labels = labels.reindex(oof_proba.index)
    proba = oof_proba[list(LABELS)]
    best, argmax_f1, best_f1 = _search(lambda d: _macro_f1(d, proba, labels))
    best.fitted_on_clients = [str(c) for c in proba.index]
    return best, {"argmax_macro_f1": argmax_f1, "tuned_macro_f1": best_f1}


def expected_macro_f1(predicted: pd.Series, proba: pd.DataFrame) -> float:
    """Macro-F1 expected on a batch when its own probabilities stand in as soft labels. For each
    label c: E[TP_c] = sum of p_ic over the Clients predicted c, E[true_c] = sum_i p_ic, and
    F1_c = 2 E[TP_c] / (#predicted c + E[true_c]) (0 when both are 0); the mean over the eight
    labels. No labels are read."""
    p = proba[list(LABELS)].to_numpy(dtype=float)
    hot = np.asarray(predicted.reindex(proba.index), dtype=object)[:, None] == np.array(LABELS, dtype=object)[None, :]
    tp = (p * hot).sum(axis=0)
    denom = hot.sum(axis=0) + p.sum(axis=0)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(f1.mean())


def fit_expected(proba: pd.DataFrame) -> tuple[Decision, dict[str, float]]:
    """The batch decision D: E3's search, over E3's grid and parameters, maximising the batch's
    own expected macro-F1 (`expected_macro_f1`) instead of macro-F1 on labels. Unsupervised: it
    is picked on the very Clients it then labels."""
    proba = proba[list(LABELS)]
    best, argmax_f1, best_f1 = _search(lambda d: expected_macro_f1(d.apply(proba), proba))
    best.fitted_on_clients = [str(c) for c in proba.index]
    return best, {"argmax_expected_macro_f1": argmax_f1, "expected_macro_f1": best_f1}
