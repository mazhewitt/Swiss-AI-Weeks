"""Is the filler signal learnable where the base rate matches test? Cross-validation over train plus the
selection set (`data.training_run(with_selection=True)`, the refit's own footing), scored on the selection
Clients only; and cross-validation over the selection set alone. Variants as in filler_survival.py."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "scripts"); sys.path.insert(0, "experiments/analysis/none_probe")
from none_model_variants import macro_f1s
from filler_survival import VARIANTS, filler_table, fit_predict
from recurring_family import data
from recurring_family.cli import Paths, _training_data
from recurring_family.config import LABELS
from recurring_family.cross_validation import CV_FOLDS, CV_SEED

OUT = Path("experiments/analysis/none_probe")


def oof(table, labels, feats):
    clients = labels.index
    parts = []
    for fold, (fit_idx, score_idx) in enumerate(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=CV_SEED).split(clients, labels.to_numpy())):
        fit_c, score_c = set(map(str, clients[fit_idx])), clients[score_idx]
        tt = table[table["client_id"].isin(fit_c)].reset_index(drop=True)
        ts = table[table["client_id"].isin(set(map(str, score_c)))].reset_index(drop=True)
        proba, _ = fit_predict(tt, labels.iloc[fit_idx], ts, score_c, feats)
        proba.insert(0, "fold", fold); parts.append(proba)
    return pd.concat(parts).reindex(clients).rename_axis("client_id")


def main():
    paths = Paths(Path("."))
    results = {}
    with data.training_run(with_selection=True):
        tx, labels = _training_data(paths, with_selection=True)
        train_labels = data.load_labels(paths.raw, "train").set_index("client_id").iloc[:, 0]
        selection = labels.index.difference(train_labels.index)
        table = filler_table(tx, labels.index)
        for name, feats in VARIANTS.items():
            o = oof(table, labels, feats)
            on_sel = macro_f1s(o.loc[selection], labels.loc[selection])
            sel_only = macro_f1s(oof(table[table["client_id"].isin(set(selection))].reset_index(drop=True), labels.loc[selection], feats), labels.loc[selection])
            results[name] = {"train+selection cv, scored on selection": on_sel, "selection-only cv": sel_only}
            print(name, "| cv over train+selection, selection Clients:", json.dumps(on_sel), "| cv over selection alone:", json.dumps(sel_only), flush=True)
    (OUT / "filler_on_selection.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
