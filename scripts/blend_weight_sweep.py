"""Choose the blend's rule weight on train out-of-fold probabilities (ticket 07). Reads no valid labels.

The gated rule and the Stream Ranker (with the Pseudo-Label sources of the Day-2 12:00 upload) are each
cross-validated once on the train Clients with the same folds; the blend's out-of-fold probabilities at
rule weight w are then exactly w * rule + (1 - w) * ranker. For every w the tuned decision layer (E3) is
fitted on those rows, and its macro-F1 on them is reported with the argmax one. The tuned figure is
fitted and scored on the same rows, so it is optimistic by about the same amount for every w: use it to
rank the weights, not as an estimate of the selection score.

    uv run python scripts/blend_weight_sweep.py [--out experiments/blend_weight_sweep.csv]
"""

import argparse
from pathlib import Path

import pandas as pd

from recurring_family import data
from recurring_family import decision as decision_layer
from recurring_family.cli import Paths, _new_model, _pseudo_training, _training_data
from recurring_family.config import LABELS, SHIFTED_CUTOFF
from recurring_family.cross_validation import out_of_fold_proba

WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="experiments/blend_weight_sweep.csv")
    args = parser.parse_args()
    paths = Paths(Path(args.root))

    settings = argparse.Namespace(
        model="blend", param=[], ordering=None, none_gate=4, rule_weight=None,
        pseudo=[("train", SHIFTED_CUTOFF), ("unlabeled", SHIFTED_CUTOFF)], pseudo_weight=None, pseudo_min_payments=None,
    )
    pseudo, info = _pseudo_training(settings, paths)
    with data.training_run():
        transactions, labels = _training_data(paths, with_selection=False)
        rule = out_of_fold_proba(lambda: _new_model(settings, pseudo, info).rules, transactions, labels)[list(LABELS)]
        ranker = out_of_fold_proba(lambda: _new_model(settings, pseudo, info).ranker, transactions, labels)[list(LABELS)]

    rows = []
    for w in WEIGHTS:
        blended = w * rule + (1.0 - w) * ranker
        _, fit = decision_layer.fit(blended, labels)
        rows.append({"rule_weight": w, "argmax_macro_f1": round(fit["argmax_macro_f1"], 4),
                     "tuned_macro_f1": round(fit["tuned_macro_f1"], 4)})
        print(f"rule weight {w:.1f}: argmax {rows[-1]['argmax_macro_f1']:.4f}, tuned {rows[-1]['tuned_macro_f1']:.4f}")
    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    best = table.loc[table["tuned_macro_f1"].idxmax()]
    print(f"best rule weight on train out-of-fold (tuned): {best['rule_weight']:g} -> {args.out}")


if __name__ == "__main__":
    main()
