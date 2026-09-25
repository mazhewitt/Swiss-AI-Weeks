"""Ticket 19, stage 1: the train-only base B (hail mary A's two members) predicting the test Clients.

Fits each member on train only, with hail mary's own code, factories and seeds (its `fit rehearsal <model>`
stage), and predicts both the 700 selection Clients and the 1,000 test Clients from the same fitted model.
The selection predictions must equal `artifacts/hailmary/rehearsal/target_<model>.csv` exactly; the test
predictions are B's input to the stacker on test. No valid label is read here.

    uv run python experiments/analysis/none_stack/base_test.py <v2|surv>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "experiments/analysis/hailmary")
import hailmary as hm  # noqa: E402
from recurring_family import data  # noqa: E402
from recurring_family.config import LABELS  # noqa: E402

CACHE = hm.PATHS.artifacts / "none_stack"


def main(model: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    make = hm.FACTORIES[model]()
    tx, labels = hm.labelled("rehearsal")  # train only
    with data.training_run(with_selection=False):
        fitted = make().fit(tx, labels)
    for domain, name in (("rehearsal", "selection"), ("final", "test")):
        target_tx, clients = hm.target(domain)
        with data.predicting():
            proba = fitted.predict_proba(target_tx, clients)[list(LABELS)]
        path = CACHE / f"base_{model}_{name}.csv"
        proba.to_csv(path, index_label="client_id")
        print(f"{model}: predicted {len(proba)} {name} Clients -> {path}", flush=True)
    # sanity check: the selection predictions reproduce hail mary's rehearsal cache exactly
    ours = (CACHE / f"base_{model}_selection.csv").read_bytes()
    theirs = (hm.cache("rehearsal") / f"target_{model}.csv").read_bytes()
    same = ours == theirs
    if not same:
        a = hm.read_proba(CACHE / f"base_{model}_selection.csv")
        b = hm.read_proba(hm.cache("rehearsal") / f"target_{model}.csv")
        print(f"{model}: NOT byte-identical; max abs diff {float((a - b.reindex(a.index)).abs().max().max()):.3g}", flush=True)
        raise SystemExit(1)
    print(f"{model}: selection predictions byte-identical to hail mary's rehearsal cache", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
