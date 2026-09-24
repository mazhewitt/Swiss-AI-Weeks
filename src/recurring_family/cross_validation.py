"""Stratified k-fold cross-validation producing out-of-fold probabilities for any model."""

from typing import Callable

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import LABELS

CV_FOLDS = 5
CV_SEED = 0


def out_of_fold_proba(
    make_model: Callable[[], object],
    transactions: pd.DataFrame,
    labels: pd.Series,
    folds: int = CV_FOLDS,
) -> pd.DataFrame:
    """One row per labelled Client (in `labels` order): its fold and the probabilities
    from a fresh model fitted on the other folds only."""
    clients = labels.index
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=CV_SEED)
    parts = []
    for fold, (fit_idx, score_idx) in enumerate(splitter.split(clients, labels.to_numpy())):
        fit_clients, score_clients = clients[fit_idx], clients[score_idx]
        model = make_model().fit(
            transactions[transactions["client_id"].isin(set(fit_clients))], labels.iloc[fit_idx]
        )
        proba = model.predict_proba(
            transactions[transactions["client_id"].isin(set(score_clients))], score_clients
        )[list(LABELS)]
        proba.insert(0, "fold", fold)
        parts.append(proba)
    oof = pd.concat(parts).reindex(clients)
    oof.index.name = "client_id"
    return oof
