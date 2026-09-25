"""Ticket 14 step 2: the soft race on train out-of-fold (folds of `rf cv`). Reads no valid labels.

Pre-registered variants, no others: A soft-predict (`soft=True`: the hard fit, so s is V2's; only the race
changes) and B soft-fit (`soft_fit=True`: soft-weighted rows and the soft race), against V2 `monthly-slot`
and V0 `unprojected-last`.
Diagnostics 1-5 of the ticket. Results in `soft_race.json`, out-of-fold probabilities in `oof_soft_*.csv`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "scripts")
from none_model_variants import macro_f1s  # noqa: E402
from recurring_family import data, decision as decision_layer
from recurring_family.cli import Paths, _training_data
from recurring_family.config import LABELS, NONE_LABEL
from recurring_family.cross_validation import CV_FOLDS, CV_SEED
from recurring_family.evaluation import paired_bootstrap, score
from recurring_family.models import predict_labels
from recurring_family.ranker import _streams_of
from recurring_family.survival import SurvivalModel, race_slot, race_table

OUT = Path("experiments/analysis/survival")
CLOSE_DAYS = 3.0


def nested_predictions(oof: pd.DataFrame, labels: pd.Series) -> pd.Series:
    proba = oof[list(LABELS)]
    nested = pd.Series(index=proba.index, dtype=object)
    for fold in sorted(oof["fold"].unique()):
        fit = oof["fold"] != fold
        decision, _ = decision_layer.fit(proba[fit], labels[fit])
        nested[~fit] = decision.apply(proba[~fit])
    return nested


def main():
    paths = Paths(Path("."))
    with data.training_run():
        transactions, labels = _training_data(paths, with_selection=False)
        clients = labels.index
        splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
        oofs = {k: [] for k in ("V0", "V2", "A", "B")}
        stream_rows = []  # per scored Client under V2's fold model: its streams' s, slot, family
        for fold, (fit_idx, score_idx) in enumerate(splitter.split(clients, labels.to_numpy())):
            fit_c, score_c = clients[fit_idx], clients[score_idx]
            tx_fit = transactions[transactions["client_id"].isin(set(fit_c))]
            tx_score = transactions[transactions["client_id"].isin(set(score_c))]
            v2 = SurvivalModel(order="monthly-slot").fit(tx_fit, labels.iloc[fit_idx])
            a = SurvivalModel(order="monthly-slot", soft=True)  # the same fit as V2: only the race changes
            a.booster, a.constant = v2.booster, v2.constant
            b = SurvivalModel(order="monthly-slot", soft=True, soft_fit=True).fit(tx_fit, labels.iloc[fit_idx])
            v0 = SurvivalModel(order="unprojected-last").fit(tx_fit, labels.iloc[fit_idx])
            for name, model in (("V0", v0), ("V2", v2), ("A", a), ("B", b)):
                p = model.predict_proba(tx_score, score_c)[list(LABELS)]
                p.insert(0, "fold", fold)
                oofs[name].append(p)
            table = race_table(_streams_of(tx_score, score_c), order="monthly-slot")
            stream_rows.append(table.assign(s=v2.survival(table), slot=race_slot(table, "monthly-slot").to_numpy()))
            print(f"fold {fold}: B fit {b.summary()}", flush=True)
        oofs = {k: pd.concat(v).reindex(clients).rename_axis("client_id") for k, v in oofs.items()}
        streams = pd.concat(stream_rows, ignore_index=True)
        full = race_table(_streams_of(transactions, clients), order="monthly-slot")

    results = {}
    base_argmax = predict_labels(oofs["V2"][list(LABELS)])
    nested = {k: nested_predictions(o, labels) for k, o in oofs.items()}
    for name, oof in oofs.items():
        oof.to_csv(OUT / f"oof_soft_{name}.csv")
        argmax = predict_labels(oof[list(LABELS)])
        row = {**macro_f1s(oof, labels), "argmax_changed_vs_V2": int((argmax != base_argmax).sum())}
        if name != "V2":
            bs = paired_bootstrap(labels, nested[name], nested["V2"])
            row["nested_vs_V2"] = {k: round(float(v), 4) for k, v in bs.items()}
        results[name] = row
        print(name, json.dumps(row), flush=True)

    # diagnostic 3: P(none) is order-free, so A's none equals V2's
    diff = float((oofs["A"][NONE_LABEL] - oofs["V2"][NONE_LABEL]).abs().max())
    results["diag3_max_abs_none_diff_A_vs_V2"] = diff
    print(f"diag 3: max |P(none) A - V2| = {diff:.2e}")

    # diagnostic 1: the close-call ceiling under V2 argmax
    streams["client_id"] = streams["client_id"].astype(str)
    streams["family"] = streams["family"].astype(str)
    first = streams.sort_values(["client_id", "order"]).groupby(["client_id", "family"]).first()
    truth = labels.astype(str)
    truth.index = truth.index.astype(str)
    pred = base_argmax.astype(str)
    pred.index = pred.index.astype(str)
    wrong_family = [c for c in truth.index if truth[c] != NONE_LABEL and pred[c] != truth[c] and pred[c] != NONE_LABEL
                    and (c, truth[c]) in first.index]
    d1 = []
    for c in wrong_family:
        t, p = first.loc[(c, truth[c])], first.loc[(c, pred[c])]
        d1.append(dict(client_id=c, gap=abs(t["slot"] - p["slot"]), s_true_higher=bool(t["s"] > p["s"]),
                       true_first=bool(t["order"] < p["order"])))
    d1 = pd.DataFrame(d1)
    close = d1[d1["gap"] <= CLOSE_DAYS]
    within7 = d1[d1["gap"] <= 7]
    results["diag1"] = {
        "V2_argmax_wrong_detected_family": int(len(d1)),
        "within_3_days": int(len(close)), "within_3_days_and_s_true_higher": int(close["s_true_higher"].sum()),
        "within_7_days": int(len(within7)), "within_7_days_and_s_true_higher": int(within7["s_true_higher"].sum()),
        "any_gap_s_true_higher": int(d1["s_true_higher"].sum()),
        "true_stream_raced_first": int(d1["true_first"].sum()),
        "gap_quantiles": {str(q): round(float(d1["gap"].quantile(q)), 1) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
    }
    print("diag 1:", json.dumps(results["diag1"]))

    # diagnostic 2: net flips among close-call Clients (argmax and nested tuned), and overall
    close_ids = set(close["client_id"])
    for name in ("A", "B"):
        for kind, preds in (("argmax", predict_labels(oofs[name][list(LABELS)])), ("nested", nested[name])):
            base = base_argmax if kind == "argmax" else nested["V2"]
            preds, base = preds.astype(str), base.astype(str)
            preds.index, base.index = preds.index.astype(str), base.index.astype(str)
            fixed = (preds == truth) & (base != truth)
            broken = (preds != truth) & (base == truth)
            in_close = pd.Series(truth.index.isin(close_ids), index=truth.index)
            results[f"diag2_{name}_{kind}"] = {
                "fixed": int(fixed.sum()), "broken": int(broken.sum()),
                "fixed_close_call": int((fixed & in_close).sum()), "broken_close_call": int((broken & in_close).sum()),
            }
            print(f"diag 2 {name} {kind}:", json.dumps(results[f"diag2_{name}_{kind}"]))

    # diagnostic 5: active streams that are overdue and rolled a period forward
    active = full[full["active"] == 1.0]
    rolled = active["days_since_last"] > active["period_days"]
    results["diag5"] = {"active_streams": int(len(active)), "overdue_and_rolled": int(rolled.sum()),
                        "share": round(float(rolled.mean()), 3),
                        "clients_with_one": int(active.loc[rolled, "client_id"].nunique()), "clients_with_active": int(active["client_id"].nunique())}
    print("diag 5:", json.dumps(results["diag5"]))
    (OUT / "soft_race.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
