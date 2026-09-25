"""Ticket 21: the hail mary's configuration A with the labelled valid (selection) Clients at weight 3.

Weight 3 is implemented as duplicates: two extra copies of each labelled selection Client in the fit, with
ids `dup1:<id>` and `dup2:<id>` and their transaction rows copied under the new ids (as the hail mary's
`self:` Clients). The decision layer (E3) is always fitted on the out-of-fold probabilities of the
weight-1 fit, so a duplicate never enters a cross-validation split. The v2 ranker's internal `none`
cross-fit is made group-aware (a Client's copies share a fold); with no copies it is the original fit.

Rehearsal: the 700 selection Clients in two fixed halves (seed 0, stratified by label). For each half h,
v2 and the Survival Race are fitted on train + the other half (weight 1, and weight 3), E3 on the weight-1
out-of-fold probabilities of train + the other half, and half h is predicted. Final: train + all 700
selection Clients; E3 is the hail mary's final one (its weight-1 out-of-fold files are reused).

Stages (cached under artifacts/target_weight/, so they run as separate processes in parallel):

    halves                    the two fixed halves (reads selection labels inside a training run)
    oof <h0|h1> <v2|surv>     weight-1 5-fold out-of-fold probabilities on train + the other half
    fit <h0|h1|final> <1|3> <v2|surv>
                              fit on the domain's labelled Clients at that weight, predict the target
    predict                   freeze both rehearsal prediction sets; write the final files and checks
    score                     the only stage that reads the selection labels to score (data.scoring())

    uv run python experiments/analysis/target_weight/target_weight.py <stage> ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hailmary"))
import hailmary as hm  # noqa: E402

from recurring_family import data  # noqa: E402
from recurring_family import decision as decision_layer  # noqa: E402
from recurring_family.config import LABEL_COLUMN, LABELS  # noqa: E402
from recurring_family.cross_validation import out_of_fold_proba  # noqa: E402
from recurring_family.evaluation import paired_bootstrap, score  # noqa: E402
from recurring_family.none_model import NoneModel  # noqa: E402
from recurring_family.config import NONE_LABEL  # noqa: E402
from recurring_family.ranker import NONE_FOLDS, RANKER_NONE, RankerModel  # noqa: E402
from recurring_family.submission import write_submission  # noqa: E402

PATHS = hm.PATHS
OUT = Path("experiments") / "analysis" / "target_weight"
CACHE = PATHS.artifacts / "target_weight"
SEED = 0
SUBMISSION = "day2_final_target_weight"
V2_FILE = PATHS.submissions / "day2_final_ranker_none_v2.csv"
DOMAINS = ("h0", "h1", "final")
WEIGHTS = (1, 3)
_DUP = re.compile(r"^dup\d+:")


def cache(domain: str) -> Path:
    path = CACHE / domain
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- models -------------------------------------------------------------------------------------------


def _base(client: str) -> str:
    return _DUP.sub("", str(client))


class GroupedRanker(RankerModel):
    """The ranker with its `none` cross-fit folds drawn over base Clients, so a Client's duplicates share
    its fold. With no duplicates the folds are exactly the original ones (same KFold over the same list)."""

    def _cross_fitted_none(self, transactions: pd.DataFrame, labels: pd.Series) -> pd.Series:
        clients = pd.Index(labels.index.astype(str), name="client_id")
        out = pd.Series(np.nan, index=clients, name=RANKER_NONE)
        if len(clients) < 2:
            return out
        bases = pd.Index([_base(c) for c in clients])
        unique = pd.Index(pd.unique(bases))
        splitter = KFold(n_splits=min(NONE_FOLDS, len(unique)), shuffle=True, random_state=0)
        for fit_b, score_b in splitter.split(unique):
            fit_idx = np.flatnonzero(bases.isin(set(unique[fit_b])))
            score_idx = np.flatnonzero(bases.isin(set(unique[score_b])))
            ranker = RankerModel(pseudo=self.pseudo, pseudo_weight=self.pseudo_weight)
            ranker.fit(transactions, labels.iloc[fit_idx])
            out.iloc[score_idx] = ranker.predict_proba(transactions, labels.index[score_idx])[NONE_LABEL].to_numpy()
        return out


def v2_factory():
    """hailmary.v2_factory's model (same Pseudo-Labels), with the group-aware `none` cross-fit."""
    hm.v2_factory()
    pseudo, info = hm._PSEUDO
    return lambda: GroupedRanker(pseudo=pseudo, pseudo_weight=info["weight"], none_model=NoneModel())


FACTORIES = {"v2": v2_factory, "surv": hm.surv_factory}


# --- data ---------------------------------------------------------------------------------------------


def stage_halves() -> None:
    with data.training_run(with_selection=True):
        sel = data.load_labels(PATHS.raw, "selection")[["client_id", LABEL_COLUMN]]
    sel = sel.assign(client_id=sel["client_id"].astype(str)).sort_values("client_id").reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    half = np.zeros(len(sel), dtype=int)
    for h, (_, idx) in enumerate(splitter.split(sel["client_id"], sel[LABEL_COLUMN])):
        half[idx] = h
    out = pd.DataFrame({"client_id": sel["client_id"], "half": half})  # ids only, no labels
    out.to_csv(CACHE / "halves.csv", index=False)
    print(f"halves: {np.bincount(half).tolist()}", flush=True)


def halves() -> dict[int, set[str]]:
    df = pd.read_csv(CACHE / "halves.csv", dtype={"client_id": str})
    return {h: set(df.loc[df["half"] == h, "client_id"]) for h in (0, 1)}


def fit_set(domain: str, weight: int) -> tuple[pd.DataFrame, pd.Series]:
    """The domain's labelled Clients, with each selection Client counted `weight` times. Labels are read
    inside a training run only; in a rehearsal half, the held-out half's are dropped on load."""
    with data.training_run(with_selection=True):
        if domain == "final":
            tx, labels = hm._training_data(PATHS, with_selection=True)
            sel_ids = set(data.load_labels(PATHS.raw, "selection")["client_id"].astype(str))
        else:
            other = halves()[1 - int(domain[1])]
            labels = data.load_labels(PATHS.raw, "train").set_index("client_id")[LABEL_COLUMN]
            sel = data.load_labels(PATHS.raw, "selection")
            sel = sel[sel["client_id"].astype(str).isin(other)].set_index("client_id")[LABEL_COLUMN]
            tx = data.load_transactions(PATHS.raw, "train")
            valid_tx = data.load_transactions(PATHS.raw, "valid")
            labels = pd.concat([labels, sel])
            tx = pd.concat([tx, valid_tx[valid_tx["client_id"].astype(str).isin(set(sel.index.astype(str)))]], ignore_index=True)
            sel_ids = set(sel.index.astype(str))
    if weight == 1:
        return tx, labels
    labels_s = labels.astype(str)
    sel_labels = labels_s[labels_s.index.astype(str).isin(sel_ids)]
    sel_tx = tx[tx["client_id"].astype(str).isin(sel_ids)]
    tx_parts, label_parts = [tx], [labels_s]
    for k in range(1, weight):
        prefix = f"dup{k}:"
        copy = sel_tx.copy()
        copy["client_id"] = (prefix + copy["client_id"].astype(str)).astype(tx["client_id"].dtype)
        tx_parts.append(copy)
        label_parts.append(pd.Series(sel_labels.to_numpy(), index=pd.Index(prefix + sel_labels.index.astype(str), name="client_id")))
    out_labels = pd.concat(label_parts)
    out_labels.name = labels.name
    print(f"fit set {domain} w={weight}: {len(labels)} Clients + {len(out_labels) - len(labels)} duplicates", flush=True)
    return pd.concat(tx_parts, ignore_index=True), out_labels


def target(domain: str) -> tuple[pd.DataFrame, pd.Index]:
    if domain == "final":
        return hm.target("final")
    ids = sorted(halves()[int(domain[1])])
    tx = data.load_transactions(PATHS.raw, "valid")
    tx = tx[tx["client_id"].astype(str).isin(set(ids))].reset_index(drop=True)
    return tx, pd.Index(ids, name="client_id")


# --- stages -------------------------------------------------------------------------------------------


def stage_oof(domain: str, model: str) -> None:
    make = FACTORIES[model]()
    tx, labels = fit_set(domain, 1)
    with data.training_run(with_selection=True):
        oof = out_of_fold_proba(make, tx, labels)
    oof.to_csv(cache(domain) / f"oof_{model}.csv")
    print(f"oof {domain} {model}: {len(oof)} Clients", flush=True)


def stage_fit(domain: str, weight: int, model: str) -> None:
    make = FACTORIES[model]()
    tx, labels = fit_set(domain, weight)
    with data.training_run(with_selection=True):
        fitted = make().fit(tx, labels)
    target_tx, clients = target(domain)
    with data.predicting():
        proba = fitted.predict_proba(target_tx, clients)[list(LABELS)]
    proba.to_csv(cache(domain) / f"target_{model}_w{weight}.csv", index_label="client_id")
    print(f"fit {domain} w={weight} {model}: predicted {len(proba)} target Clients", flush=True)


def averaged(domain: str, weight: int) -> pd.DataFrame:
    v2 = hm.read_proba(cache(domain) / f"target_v2_w{weight}.csv")
    surv = hm.read_proba(cache(domain) / f"target_surv_w{weight}.csv")
    return (v2 + surv.reindex(v2.index)) / 2


def rehearsal_decision(domain: str) -> decision_layer.Decision:
    oof = (hm.read_proba(cache(domain) / "oof_v2.csv") + hm.read_proba(cache(domain) / "oof_surv.csv")) / 2
    _, labels = fit_set(domain, 1)
    labels = pd.Series(labels.astype(str).to_numpy(), index=labels.index.astype(str))
    decision, _ = decision_layer.fit(oof, labels.reindex(oof.index))
    return decision


def stage_predict() -> None:
    """Freeze both rehearsal prediction sets (no selection label is read for scoring here), and write the
    final files: the w = 3 submission, and the w = 1 check against the hail mary's configuration A."""
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    decisions = {d: rehearsal_decision(d) for d in ("h0", "h1")}
    for w in WEIGHTS:
        parts = [decisions[d].apply(averaged(d, w)) for d in ("h0", "h1")]
        pred = pd.concat(parts).sort_index()
        assert len(pred) == 700 and pred.notna().all() and pred.index.is_unique
        pred.rename("predicted").to_csv(OUT / "predictions" / f"rehearsal_w{w}.csv", index_label="client_id")
    # final: E3 of the hail mary's final (weight-1 out-of-fold probabilities of configuration A)
    decision, _ = hm.decision_and_prior("final")
    expected = data.load_sample_submission(PATHS.raw)["client_id"]
    final = {w: decision.apply(averaged("final", w)) for w in WEIGHTS}
    a_proba = hm.averaged("final")
    a_pred = decision.apply(a_proba)
    w1_proba = averaged("final", 1).reindex(a_proba.index)
    checks = {
        "final_w1_vs_hailmary_A_label_agreement": float((final[1].reindex(a_pred.index) == a_pred).mean()),
        "final_w1_vs_hailmary_A_max_abs_proba_diff": float((w1_proba - a_proba).abs().to_numpy().max()),
    }
    write_submission(final[3], expected, PATHS.submissions / f"{SUBMISSION}.csv")
    for w in WEIGHTS:
        final[w].reindex(expected).rename("predicted").to_csv(OUT / "predictions" / f"final_test_w{w}.csv", index_label="client_id")
    v2 = pd.read_csv(V2_FILE, dtype=str, keep_default_na=False).set_index("client_id").iloc[:, 0].reindex(expected)
    w3 = final[3].reindex(expected)
    info = {
        **checks,
        "final_w3_disagree_v2_file": float((w3 != v2).mean()),
        "final_w1_disagree_v2_file": float((final[1].reindex(expected) != v2).mean()),
        "final_w3_disagree_w1": float((w3 != final[1].reindex(expected)).mean()),
        "final_w3_label_mix": {k: int(v) for k, v in w3.value_counts().items()},
        "final_written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT / "final_checks.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def stage_score() -> None:
    preds = {
        w: pd.read_csv(OUT / "predictions" / f"rehearsal_w{w}.csv", dtype=str, keep_default_na=False)
        .set_index("client_id")["predicted"]
        for w in WEIGHTS
    }
    with data.scoring():
        truth = data.load_labels(PATHS.raw, "selection").set_index("client_id")[LABEL_COLUMN]
    truth.index = truth.index.astype(str)
    truth = truth.astype(str)
    assert all(set(p.index) == set(truth.index) for p in preds.values())
    s = {w: score(truth, preds[w].reindex(truth.index)) for w in WEIGHTS}
    cmp = paired_bootstrap(truth, preds[3].reindex(truth.index), preds[1].reindex(truth.index))
    final = json.loads((OUT / "final_checks.json").read_text())
    rule1 = s[3]["macro_f1"] - s[1]["macro_f1"] > 0
    rule2 = final["final_w3_disagree_v2_file"] >= 0.10
    result = {
        "rehearsal": {
            f"w{w}": {"macro_f1": round(s[w]["macro_f1"], 4), **{f"f1_{l}": round(s[w][f"f1_{l}"], 4) for l in LABELS}}
            for w in WEIGHTS
        },
        "w3_vs_w1_paired_bootstrap": {k: round(v, 4) for k, v in cmp.items()},
        "rehearsal_agreement_w3_w1": round(float((preds[3] == preds[1].reindex(preds[3].index)).mean()), 4),
        "final": final,
        "rule": {
            "w3_beats_w1_pooled": bool(rule1),
            "w3_disagrees_v2_file_ge_10pct": bool(rule2),
            "ready_and_checked_by_1710_cest": None,  # filled in after `rf submit --check`
        },
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="stage", required=True)
    sub.add_parser("halves")
    s = sub.add_parser("oof")
    s.add_argument("domain", choices=("h0", "h1"))
    s.add_argument("model", choices=sorted(FACTORIES))
    s = sub.add_parser("fit")
    s.add_argument("domain", choices=DOMAINS)
    s.add_argument("weight", type=int, choices=WEIGHTS)
    s.add_argument("model", choices=sorted(FACTORIES))
    sub.add_parser("predict")
    sub.add_parser("score")
    a = p.parse_args()
    if a.stage == "halves":
        stage_halves()
    elif a.stage == "oof":
        stage_oof(a.domain, a.model)
    elif a.stage == "fit":
        stage_fit(a.domain, a.weight, a.model)
    elif a.stage == "predict":
        stage_predict()
    else:
        stage_score()


if __name__ == "__main__":
    main()
