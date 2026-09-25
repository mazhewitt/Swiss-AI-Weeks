"""argmax, tuned and nested tuned macro-F1 of `rf cv` out-of-fold files (train labels only). Nested: the
decision layer fitted on four folds' rows and applied to the fifth, via the `fold` column.

    uv run python experiments/analysis/filler_streams/oof_score.py NAME=OOF.csv ...
"""
import sys
from pathlib import Path
import pandas as pd
from recurring_family import data
from recurring_family import decision as decision_layer
from recurring_family.config import LABELS
from recurring_family.evaluation import score

with data.training_run():
    labels = data.load_labels(Path("data/raw"), "train").set_index("client_id")["target_next_recurring_merchant"]
rows = []
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    oof = pd.read_csv(path, index_col="client_id")
    y = labels.reindex(oof.index)
    proba = oof[list(LABELS)]
    _, fitted = decision_layer.fit(proba, y)
    nested = pd.Series(index=proba.index, dtype=object)
    for fold in sorted(oof["fold"].unique()):
        fit = oof["fold"] != fold
        decision, _ = decision_layer.fit(proba[fit], y[fit])
        nested[~fit] = decision.apply(proba[~fit])
    rows.append({"run": name, "argmax": round(fitted["argmax_macro_f1"], 4), "tuned": round(fitted["tuned_macro_f1"], 4),
                 "nested_tuned": round(score(y, nested)["macro_f1"], 4)})
    print(rows[-1], flush=True)
print(pd.DataFrame(rows).to_string(index=False))
