"""Ticket 18, post hoc (not pre-registered; context only, no bearing on the gate). Reads the cached feature
blocks and out-of-fold labels of `diagnostic.py`; no label file is read here.

1. Per-feature drift of the top 10 raw features: single-feature AUC of selection vs test and train vs test
   (label-free, direction-free), and the three domain means.
2. The raw block's `none` AUC with its Decoy features removed (same repeated 5x5 CV), to see how much of the
   gate rests on the one feature that shifts between selection and test.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "experiments/analysis/target_none")
from diagnostic import auc_ci, paired, repeated_oof  # noqa: E402

OUT = Path("experiments/analysis/target_none")
CACHE = OUT / "cache"


def single_auc(a: pd.Series, b: pd.Series) -> float:
    x = pd.concat([a, b])
    auc = roc_auc_score([0] * len(a) + [1] * len(b), x.fillna(x.median()))
    return round(max(auc, 1 - auc), 4)


def main():
    results = json.loads((OUT / "results.json").read_text())
    sel, test, train = (pd.read_pickle(CACHE / s / "raw.pkl") for s in ("selection", "test", "train"))
    drift = {}
    for f in results["adversarial_top10_raw"]["features"]:
        drift[f] = {"selection_vs_test": single_auc(sel[f], test[f]), "train_vs_test": single_auc(train[f], test[f]),
                    "mean_train": round(float(train[f].mean()), 4), "mean_selection": round(float(sel[f].mean()), 4),
                    "mean_test": round(float(test[f].mean()), 4)}
        print(f, drift[f], flush=True)

    oof = pd.read_csv(CACHE / "oof_scores.csv", index_col=0, dtype={0: str})
    y = oof["none"].reindex(sel.index.astype(str)).astype(int)
    y.index = sel.index
    decoy = [c for c in sel.columns if "decoy" in c]
    s, gain, reps = repeated_oof(sel.drop(columns=decoy), y, "raw-decoy")
    raw_full = oof["raw"].reindex(sel.index.astype(str)).to_numpy()
    results["posthoc"] = {
        "note": "not pre-registered; context only",
        "top10_raw_per_feature_drift": drift,
        "raw_without_decoy_features": {**auc_ci(y.to_numpy(), s.to_numpy()), "dropped": decoy, "per_repeat_auc": reps,
                                       "top_gain": gain.head(10).round(0).to_dict(),
                                       "paired_vs_raw": paired(y.to_numpy(), s.to_numpy(), raw_full)},
    }
    print(json.dumps(results["posthoc"]["raw_without_decoy_features"], indent=1))
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
