"""Ticket 13 fix round 2: the race order among unprojected (single-payment) streams, on train out-of-fold
(folds of `rf cv`). Reads no valid labels. Pre-registered variants, no others:

V0 `unprojected-last` (the order so far), V1 `recent-first`, V2 `monthly-slot` (see `survival.ORDERS`).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scripts")
from none_model_variants import macro_f1s  # noqa: E402
from recurring_family import data
from recurring_family.cli import Paths, _training_data
from recurring_family.config import LABELS
from recurring_family.cross_validation import out_of_fold_proba
from recurring_family.models import predict_labels
from recurring_family.survival import ORDERS, SurvivalModel

OUT = Path("experiments/analysis/survival")


def main():
    paths = Paths(Path("."))
    with data.training_run():
        transactions, labels = _training_data(paths, with_selection=False)
        oofs = {order: out_of_fold_proba(lambda: SurvivalModel(order=order), transactions, labels) for order in ORDERS}
    base = predict_labels(oofs[ORDERS[0]][list(LABELS)])
    results = []
    for name, (order, oof) in zip(("V0", "V1", "V2"), oofs.items()):
        argmax = predict_labels(oof[list(LABELS)])
        res = {"variant": f"{name} {order}", **macro_f1s(oof, labels), "argmax_changed_vs_V0": int((argmax != base).sum())}
        print(json.dumps(res), flush=True)
        results.append(res)
    (OUT / "order_variants.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
