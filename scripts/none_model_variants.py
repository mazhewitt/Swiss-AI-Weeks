"""Choose the ranker's `none` model setup on train out-of-fold probabilities (ticket 09). Reads no valid labels.

The Stream Ranker with the Pseudo-Labels of run 20260924T171108-3251a4 (train and unlabeled at Shifted Cutoff
2025-10-03, weight 0.5, min_payments 4) is cross-validated on the train Clients with the folds of `rf cv`.
Per fold, the ranker, the `none` model's training rows (their `ranker_none` cross-fitted inside the fold)
and the scored Clients' rows are made once; each variant then fits only its `none` model on them, exactly
as `RankerModel.fit` does, so its out-of-fold rows are what `rf cv --model ranker --none-model` would give
for that feature set.

For each variant three macro-F1 figures on the 2,000 train Clients: argmax; tuned (the E3 decision fitted
and scored on the same rows, optimistic); and nested (the decision fitted on four folds' rows and applied
to the fifth), the honest one to choose by.

    uv run python scripts/none_model_variants.py [--out experiments/none_model_variants.csv]
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from recurring_family import data
from recurring_family import decision as decision_layer
from recurring_family.cli import Paths, _pseudo_training, _training_data
from recurring_family.config import LABELS, NONE_LABEL, SHIFTED_CUTOFF
from recurring_family.cross_validation import CV_FOLDS, CV_SEED
from recurring_family.evaluation import score
from recurring_family.none_model import GROUP_A, GROUP_B, NONE_PARAMS, RANKER_NONE, NoneModel, client_features, combine
from recurring_family.ranker import RankerModel, _detected

VARIANTS = {
    "A: stream fields": GROUP_A,
    "A + ranker_none": [RANKER_NONE, *GROUP_A],
    "A+B: stream fields and churn": [*GROUP_A, *GROUP_B],
    "A+B + ranker_none": [RANKER_NONE, *GROUP_A, *GROUP_B],
}
# a small grid around the analysis's hyperparameters, for the A+B variants only
GRID = {
    "leaves 15": {"num_leaves": 15},
    "min_child 60": {"min_child_samples": 60},
    "400 trees": {"n_estimators": 400},
    "lr 0.05, 300 trees": {"learning_rate": 0.05, "n_estimators": 300},
}


def fold_parts(paths: Paths, cache: Path):
    """Per fold: the ranker's probabilities for the scored Clients, the `none` model's training rows and
    the scored Clients' rows, plus the labels. Cached in `cache` (artifacts/, never committed)."""
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    settings = argparse.Namespace(
        model="ranker", pseudo=[("train", SHIFTED_CUTOFF), ("unlabeled", SHIFTED_CUTOFF)], pseudo_weight=0.5,
        pseudo_min_payments=4,
    )
    pseudo, info = _pseudo_training(settings, paths)
    with data.training_run():
        transactions, labels = _training_data(paths, with_selection=False)
    clients = labels.index
    parts = []
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    for fold, (fit_idx, score_idx) in enumerate(splitter.split(clients, labels.to_numpy())):
        fit_labels, score_clients = labels.iloc[fit_idx], clients[score_idx]
        ranker = RankerModel(pseudo=pseudo, pseudo_weight=info["weight"]).fit(transactions, fit_labels)
        proba = ranker.predict_proba(transactions, score_clients)[list(LABELS)]
        x_fit = RankerModel(pseudo=pseudo, pseudo_weight=info["weight"], none_model=NoneModel()).none_training_features(
            transactions, fit_labels
        )
        x_score = client_features(*_detected(transactions, score_clients))
        x_score.insert(0, RANKER_NONE, proba[NONE_LABEL].reindex(x_score.index))
        parts.append({"fold": fold, "proba": proba, "x_fit": x_fit, "x_score": x_score})
        print(f"fold {fold}: {len(x_fit)} none-model training Clients, {len(x_score)} scored", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps((parts, labels)))
    return parts, labels


def out_of_fold(parts, labels, features=None, params=None, active_only=False) -> pd.DataFrame:
    """The out-of-fold probabilities of the ranker (features None) or of the ranker with a `none` model."""
    rows = []
    for part in parts:
        proba = part["proba"]
        if features is not None:
            x_fit, x_score = part["x_fit"], part["x_score"]
            if active_only:
                x_fit, x_score = x_fit[x_fit["n_active_streams"] > 0], x_score[x_score["n_active_streams"] > 0]
            model = NoneModel(features=features, params=params).fit(x_fit, labels.reindex(x_fit.index) == NONE_LABEL)
            proba = combine(proba, model.predict(x_score))
        rows.append(proba.assign(fold=part["fold"]))
    return pd.concat(rows).reindex(labels.index)


def macro_f1s(oof: pd.DataFrame, labels: pd.Series) -> dict:
    proba = oof[list(LABELS)]
    _, fitted = decision_layer.fit(proba, labels)
    nested = pd.Series(index=proba.index, dtype=object)
    for fold in sorted(oof["fold"].unique()):
        fit = oof["fold"] != fold
        decision, _ = decision_layer.fit(proba[fit], labels[fit])
        nested[~fit] = decision.apply(proba[~fit])
    return {
        "argmax_macro_f1": round(fitted["argmax_macro_f1"], 4),
        "tuned_macro_f1": round(fitted["tuned_macro_f1"], 4),
        "nested_tuned_macro_f1": round(score(labels, nested)["macro_f1"], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="experiments/none_model_variants.csv")
    parser.add_argument("--oof-dir", help="also write each variant's out-of-fold probabilities here")
    args = parser.parse_args()
    paths = Paths(Path(args.root))
    parts, labels = fold_parts(paths, paths.artifacts / "none_model_variants" / "folds.pkl")

    runs = [("baseline: ranker with Pseudo-Labels", None, None, False)]
    runs += [(name, features, None, False) for name, features in VARIANTS.items()]
    runs += [(f"{name}, Active-Stream Clients only", features, None, True) for name, features in VARIANTS.items()]
    for name in ("A+B: stream fields and churn", "A+B + ranker_none"):
        runs += [(f"{name}, {g}", VARIANTS[name], {**NONE_PARAMS, **change}, False) for g, change in GRID.items()]

    rows = []
    for name, features, params, active_only in runs:
        oof = out_of_fold(parts, labels, features, params, active_only)
        if args.oof_dir:
            Path(args.oof_dir).mkdir(parents=True, exist_ok=True)
            oof[["fold", *LABELS]].to_csv(Path(args.oof_dir) / f"{len(rows):02d}.csv", index_label="client_id")
        rows.append({"variant": name, "n_features": 0 if features is None else len(features), **macro_f1s(oof, labels)})
        print(", ".join(f"{k} {v}" for k, v in rows[-1].items()), flush=True)
    table = pd.DataFrame(rows)
    base = table.loc[0, "nested_tuned_macro_f1"]
    table["nested_gain"] = (table["nested_tuned_macro_f1"] - base).round(4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
