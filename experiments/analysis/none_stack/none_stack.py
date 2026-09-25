"""Ticket 19: a target-domain `none` stacker on the hail-mary average (pre-registered in
`.scratch/ranker/issues/19-target-none-stack.md`; no other variants).

- B = hail mary A, the 1/2 v2 + 1/2 hard Survival Race average, fitted on train only (`base_test.py`, which
  reproduces hail mary's rehearsal cache exactly and also predicts the 1,000 test Clients).
- The stacker: LightGBM binary `none` on [logit P_B(none)] plus ticket 18's 130 raw features, each as its
  percentile rank within its own domain (the 700 selection Clients, or the 1,000 test Clients); ticket 18's
  fixed parameters, `n_jobs=1`, no tuning.
- P'(none) = the stacker's output; families rescaled by (1 - P'(none)) / (1 - P_B(none)) (`stacking.adjust`).
- The decision D (`decision.fit_expected`), picked label-free on the batch it labels.

Stages:

    predict   selection labels (the binary `none` target only) are read once, inside
              `data.training_run(with_selection=True)`, for the repeated 5x5 nested CV and the final stacker
              fit; writes the selection predictions of B+E3, B+D and N to predictions/ and the test
              predictions to artifacts/none_stack/
    score     reads the selection labels once, in `data.scoring()`, after every prediction is frozen; scores
              and applies the upload rule (ticket 17's pool rule, extended with N and B+D)
    submit    writes submissions/day2_final_none_stack.csv (N on test), whatever the rule chose

The sealed holdout is never read: neither its labels nor its transactions.

    uv run python experiments/analysis/none_stack/none_stack.py <stage>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "experiments/analysis/hailmary")
sys.path.insert(0, "experiments/analysis/target_none")
sys.path.insert(0, "experiments/analysis/none_stack")
import hailmary as hm  # noqa: E402
from features import blocks  # noqa: E402
from stacking import adjust, stacker_input  # noqa: E402
from recurring_family import data  # noqa: E402
from recurring_family import decision as decision_layer  # noqa: E402
from recurring_family import evaluation as ev  # noqa: E402
from recurring_family.config import LABEL_COLUMN, LABELS, NONE_LABEL  # noqa: E402
from recurring_family.evaluation import paired_bootstrap, score  # noqa: E402
from recurring_family.none_model import NONE_PARAMS  # noqa: E402
from recurring_family.submission import PREDICTION_COLUMN, write_submission  # noqa: E402

PATHS = hm.PATHS
OUT = Path("experiments/analysis/none_stack")
PRED = OUT / "predictions"
CACHE = PATHS.artifacts / "none_stack"
RAW_CACHE = Path("experiments/analysis/target_none/cache")  # ticket 18's raw-block caches (git-ignored)
PARAMS = {**NONE_PARAMS, "n_jobs": 1}  # ticket 18's fixed parameters
SEEDS = range(5)
SUBMISSION = "day2_final_none_stack"
STACK17 = Path("experiments/analysis/stack17/results.json")
HAILMARY_FILE = "day2_final_hailmary"
V2_FILE = "day2_final_ranker_none_v2"
HAILMARY_TEST_NONE = 214  # `none` rows in the committed hail-mary file
SHRINK = 1.7
MIN_DISAGREEMENT = 0.10
EPS = 1e-9


def update_results(key: str, value) -> None:
    path = OUT / "results.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    result[key] = value
    path.write_text(json.dumps(result, indent=1))


def read_labels_file(path: Path, column: str = "predicted") -> pd.Series:
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("client_id")[column]


# --- inputs (label-free) ------------------------------------------------------------------------------


def base(domain: str) -> pd.DataFrame:
    """B, the train-only hail mary A average, on `selection` or `test`."""
    v2 = hm.read_proba(CACHE / f"base_v2_{domain}.csv")
    surv = hm.read_proba(CACHE / f"base_surv_{domain}.csv")
    return (v2 + surv.reindex(v2.index)) / 2


def raw_blocks() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ticket 18's 130 raw features on selection and on test (test uses selection's MCC list, as ticket 18)."""
    sel_tx, sel_clients = hm.target("rehearsal")
    _, sel_raw, mccs = blocks(sel_tx.assign(client_id=sel_tx["client_id"].astype(str)),
                              pd.Index(sorted(sel_clients), name="client_id"), cache=RAW_CACHE / "selection")
    test_tx, test_clients = hm.target("final")
    _, test_raw, _ = blocks(test_tx.assign(client_id=test_tx["client_id"].astype(str)),
                            pd.Index(sorted(test_clients), name="client_id"), mccs=mccs, cache=RAW_CACHE / "test")
    assert sel_raw.shape == (700, 130) and test_raw.shape == (1000, 130), (sel_raw.shape, test_raw.shape)
    assert set(test_raw.index) == set(test_clients) and set(sel_raw.index) == set(sel_clients)
    assert list(sel_raw.columns) == list(test_raw.columns)
    return sel_raw, test_raw


def decision_json(d: decision_layer.Decision) -> dict:
    return {"weights": d.weights, "none_threshold": d.none_threshold}


def write_prediction(name: str, predicted: pd.Series) -> None:
    PRED.mkdir(parents=True, exist_ok=True)
    assert predicted.notna().all() and len(predicted) == 700, f"{name}: incomplete predictions"
    predicted.rename("predicted").sort_index().to_csv(PRED / f"{name}.csv", index_label="client_id")


def mix(p: pd.Series) -> dict:
    return {l: int((p == l).sum()) for l in LABELS}


# --- predict -------------------------------------------------------------------------------------------


def stage_predict() -> None:
    b_sel, b_test = base("selection"), base("test")
    sel_raw, test_raw = raw_blocks()
    x_sel = stacker_input(b_sel, sel_raw)  # ranked within the 700 selection Clients
    x_test = stacker_input(b_test, test_raw)  # ranked within the 1,000 test Clients
    assert x_sel.shape == (700, 131) and x_test.shape == (1000, 131)

    # the only label read of this stage: selection's binary `none` target, for the nested CV and the final fit
    with data.training_run(with_selection=True):
        sel = data.load_labels(PATHS.raw, "selection").set_index("client_id")[LABEL_COLUMN]
        sel.index = sel.index.astype(str)
        y = (sel.reindex(x_sel.index) == NONE_LABEL).astype(int)
        del sel
        assert len(y) == 700
        oof_total = pd.Series(0.0, index=y.index)
        per_repeat = []
        for seed in SEEDS:
            p = pd.Series(np.nan, index=y.index)
            for fit, held in StratifiedKFold(5, shuffle=True, random_state=seed).split(x_sel, y):
                m = lgb.LGBMClassifier(**PARAMS).fit(x_sel.iloc[fit], y.iloc[fit])
                p.iloc[held] = m.predict_proba(x_sel.iloc[held])[:, 1]
            assert p.notna().all()
            per_repeat.append(p)
            oof_total += p
        p_oof = oof_total / len(SEEDS)
        final = lgb.LGBMClassifier(**PARAMS).fit(x_sel, y)
        gain = pd.Series(final.booster_.feature_importance("gain"), index=x_sel.columns).sort_values(ascending=False)
        n_none = int(y.sum())
        del y
    with data.predicting():
        p_test = pd.Series(final.predict_proba(x_test)[:, 1], index=x_test.index)

    # the adjusted probabilities
    n_sel = adjust(b_sel, p_oof)
    n_test = adjust(b_test, p_test)
    for name, df in (("N selection", n_sel), ("N test", n_test)):
        assert np.allclose(df.sum(axis=1), 1.0, atol=1e-9), f"{name}: rows do not sum to 1"

    # selection candidates (label-free decisions; hail mary's E3 reads train labels only)
    e3, _ = hm.decision_and_prior("rehearsal")
    b_e3 = e3.apply(b_sel)
    committed_a = read_labels_file(hm.OUT / "predictions" / "A.csv").reindex(b_e3.index)
    same_a = float((b_e3 == committed_a).mean())
    assert same_a == 1.0, f"B+E3 does not reproduce hail mary A's committed predictions ({same_a:.3f})"
    d_b_sel, info_b_sel = decision_layer.fit_expected(b_sel)
    d_n_sel, info_n_sel = decision_layer.fit_expected(n_sel)
    selection = {"B_E3": b_e3, "B_D": d_b_sel.apply(b_sel), "N": d_n_sel.apply(n_sel)}
    for name, predicted in selection.items():
        write_prediction(name, predicted)

    # test (no test labels exist): D picked on the 1,000 test Clients' own probabilities
    d_b_test, info_b_test = decision_layer.fit_expected(b_test)
    d_n_test, info_n_test = decision_layer.fit_expected(n_test)
    test = {"B_E3": e3.apply(b_test), "B_D": d_b_test.apply(b_test), "N": d_n_test.apply(n_test)}
    for name, predicted in test.items():
        assert predicted.notna().all() and len(predicted) == 1000
        predicted.rename("predicted").sort_index().to_csv(CACHE / f"{name}_test.csv", index_label="client_id")
    p_oof.rename("p_none_oof").to_csv(CACHE / "p_none_oof_selection.csv", index_label="client_id")
    pd.DataFrame({f"seed{s}": r for s, r in zip(SEEDS, per_repeat)}).to_csv(CACHE / "p_none_oof_repeats.csv", index_label="client_id")
    n_sel.to_csv(CACHE / "N_proba_selection.csv", index_label="client_id")
    n_test.to_csv(CACHE / "N_proba_test.csv", index_label="client_id")
    b_test.to_csv(CACHE / "B_proba_test.csv", index_label="client_id")

    hailmary_file = read_labels_file(PATHS.submissions / f"{HAILMARY_FILE}.csv", PREDICTION_COLUMN)
    rnd = lambda d: {k: round(v, 4) for k, v in d.items()}  # noqa: E731
    update_results("predict", {
        "design": {"stacker_params": {k: v for k, v in PARAMS.items() if k != "verbose"},
                   "cv": "5 repeats x 5-fold stratified (seeds 0-4) on the 700 selection Clients, P'(none) averaged",
                   "features": "logit P_B(none) + 130 raw features ranked within domain", "selection_none": n_none},
        "checks": {"B+E3 vs hail mary A's committed selection predictions": same_a,
                   "N rows sum to 1": True, "degenerate P_B(none)=1 rows": {
                       "selection": int((1 - b_sel[NONE_LABEL] <= 1e-12).sum()), "test": int((1 - b_test[NONE_LABEL] <= 1e-12).sum())}},
        "stacker_top_gain": gain.head(15).round(0).to_dict(),
        "mean_p_none": {"B selection": round(float(b_sel[NONE_LABEL].mean()), 4), "N selection (oof)": round(float(p_oof.mean()), 4),
                        "B test": round(float(b_test[NONE_LABEL].mean()), 4), "N test": round(float(p_test.mean()), 4)},
        "decisions": {"selection": {"B_E3": decision_json(e3), "B_D": {**decision_json(d_b_sel), **rnd(info_b_sel)},
                                    "N": {**decision_json(d_n_sel), **rnd(info_n_sel)}},
                      "test": {"B_D": {**decision_json(d_b_test), **rnd(info_b_test)},
                               "N": {**decision_json(d_n_test), **rnd(info_n_test)}}},
        "selection_label_mix": {n: mix(p) for n, p in selection.items()},
        "test_label_mix": {**{n: mix(p) for n, p in test.items()}, "hail mary file": mix(hailmary_file)},
        "test_none": {"N": int((test["N"] == NONE_LABEL).sum()), "B_D": int((test["B_D"] == NONE_LABEL).sum()),
                      "hail mary file": int((hailmary_file == NONE_LABEL).sum())},
    })
    print(f"predict: oof P'(none) mean {p_oof.mean():.4f}; test P'(none) mean {p_test.mean():.4f}")
    print("selection mix:", json.dumps({n: mix(p) for n, p in selection.items()}))
    print("test mix:", json.dumps({n: mix(p) for n, p in test.items()}))



# --- score and the upload rule -------------------------------------------------------------------------


def paired_se(truth: pd.Series, new: pd.Series, old: pd.Series) -> float:
    """SD of the paired bootstrap deltas over the same resamples `paired_bootstrap` draws (2,000, seed 0)."""
    t, n, o = ev._one_hot(truth), ev._one_hot(new), ev._one_hot(old)
    w = ev._resample_weights(len(t))
    return float(np.std(ev._weighted_macro_f1(w, t, n) - ev._weighted_macro_f1(w, t, o), ddof=1))


def stage_score() -> None:
    names = ("B_E3", "B_D", "N")
    predictions = {n: read_labels_file(PRED / f"{n}.csv") for n in names}
    for n, p in predictions.items():
        assert p.notna().all() and len(p) == 700, f"{n}: incomplete predictions"
    tests = {n: read_labels_file(CACHE / f"{n}_test.csv") for n in ("B_D", "N")}
    p_oof = pd.read_csv(CACHE / "p_none_oof_selection.csv", dtype={"client_id": str}).set_index("client_id")["p_none_oof"]
    b_sel = base("selection")
    # every prediction is frozen; only now read the selection labels
    with data.scoring():
        truth = data.load_labels(PATHS.raw, "selection").set_index("client_id")[LABEL_COLUMN]
    truth.index = truth.index.astype(str)
    v2 = read_labels_file(PATHS.run_predictions(hm.V2_SELECTION_RUN)).reindex(truth.index)
    assert v2.notna().all()
    predictions = {n: p.reindex(truth.index) for n, p in predictions.items()}
    v2_test = read_labels_file(PATHS.submissions / f"{V2_FILE}.csv", PREDICTION_COLUMN)
    files = {"B_E3": HAILMARY_FILE, "B_D": None, "N": SUBMISSION}
    test_files = {"B_E3": read_labels_file(PATHS.submissions / f"{HAILMARY_FILE}.csv", PREDICTION_COLUMN), **tests}
    display = {"B_E3": "B+E3", "B_D": "B+D", "N": "N"}
    rows = {}
    for n in names:
        sel = predictions[n]
        s = score(truth, sel)
        vs_v2 = paired_bootstrap(truth, sel, v2)
        se = paired_se(truth, sel, v2)
        vs_b = paired_bootstrap(truth, sel, predictions["B_E3"])
        se_b = paired_se(truth, sel, predictions["B_E3"])
        disagree = float((test_files[n].reindex(v2_test.index) != v2_test).mean())
        rows[n] = {
            "candidate": display[n], "macro_f1": round(s["macro_f1"], 4),
            "d": round(vs_v2["delta"], 4), "d_low": round(vs_v2["low"], 4), "d_high": round(vs_v2["high"], 4),
            "se_paired": round(se, 4), "d_shrunk": round(vs_v2["delta"] - SHRINK * se, 4),
            "d_vs_B_E3": round(vs_b["delta"], 4), "d_vs_B_E3_low": round(vs_b["low"], 4),
            "d_vs_B_E3_high": round(vs_b["high"], 4), "se_vs_B_E3": round(se_b, 4),
            "test_disagreement_v2": round(disagree, 4), "selection_agreement_v2": round(float((sel == v2).mean()), 4),
            "none_predicted_selection": int((sel == NONE_LABEL).sum()),
            "in_pool": bool(vs_v2["delta"] > 0 and disagree >= MIN_DISAGREEMENT - EPS),
            **{f"f1_{l}": round(s[f"f1_{l}"], 4) for l in LABELS},
        }
    # check: B+E3 is hail mary A as ticket 17 scored it
    old = {r["candidate"]: r for r in json.loads(STACK17.read_text())["rehearsal"]["rows"]}
    a = old["hail mary A"]
    check = {k: (rows["B_E3"][k], a[k]) for k in ("macro_f1", "d", "se_paired", "d_shrunk", "test_disagreement_v2")}
    assert all(abs(x - y) < 1e-4 for x, y in check.values()), f"B+E3 does not match ticket 17's hail mary A row: {check}"

    # the pool: ticket 17's pool (its values), extended with N and B+D
    old_pool = json.loads(STACK17.read_text())["upload_rule"]["pool"]
    pool = [{**{k: m[k] for k in ("candidate", "d", "se_paired", "d_shrunk", "test_disagreement_v2", "file")},
             "source": "ticket 17"} for m in old_pool]
    for n in ("N", "B_D"):
        r = rows[n]
        if r["in_pool"]:
            pool.append({**{k: r[k] for k in ("candidate", "d", "se_paired", "d_shrunk", "test_disagreement_v2")},
                         "file": f"submissions/{SUBMISSION}.csv" if n == "N" else "(B+D on test: artifacts/none_stack/B_D_test.csv)",
                         "source": "ticket 19"})
    pool.sort(key=lambda m: m["d_shrunk"], reverse=True)
    chosen = pool[0] if pool else None
    rule = {
        "rule": "ticket 17's pool rule: d > 0 against v2 and >= 10% test disagreement with v2's file; ranked by "
                "d~ = d - 1.7 SE_paired; pool extended with N and B+D; the top is the 17:30 upload",
        "pool": pool, "new_not_in_pool": [display[n] for n in ("N", "B_D") if not rows[n]["in_pool"]],
        "chosen": chosen["candidate"] if chosen else None, "file": chosen["file"] if chosen else None,
        "new": bool(chosen and chosen["source"] == "ticket 19"),
    }
    y = (truth == NONE_LABEL).astype(int)
    context = {
        "p_none_auc_selection_oof": {"B": round(float(roc_auc_score(y, b_sel[NONE_LABEL].reindex(truth.index))), 4),
                                     "N (P' oof)": round(float(roc_auc_score(y, p_oof.reindex(truth.index))), 4)},
        "true_none_selection": int(y.sum()),
        "optimism": "the same 700 selection labels chose this feature block (ticket 18), so N's score is optimistic",
    }
    update_results("rehearsal", {"rows": list(rows.values()), "check_B_E3_vs_ticket17": check, "context": context})
    update_results("upload_rule", rule)
    for r in rows.values():
        print(f"{r['candidate']:<5} {r['macro_f1']:.4f}  d {r['d']:+.4f} ({r['d_low']:+.4f} .. {r['d_high']:+.4f})"
              f"  vs B+E3 {r['d_vs_B_E3']:+.4f} ({r['d_vs_B_E3_low']:+.4f} .. {r['d_vs_B_E3_high']:+.4f})"
              f"  SE {r['se_paired']:.4f}  d~ {r['d_shrunk']:+.4f}  test disagree {r['test_disagreement_v2']:.3f}  pool {r['in_pool']}")
    print(json.dumps(rule, indent=1))
    print(json.dumps(context))


# --- submit --------------------------------------------------------------------------------------------


def stage_submit() -> None:
    predicted = read_labels_file(CACHE / "N_test.csv")
    expected = data.load_sample_submission(PATHS.raw)["client_id"]
    out = PATHS.submissions / f"{SUBMISSION}.csv"
    write_submission(predicted, expected, out)
    print(f"submit: N -> {out} ({len(expected)} Clients); label mix:")
    print(predicted.value_counts().to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=("predict", "score", "submit"))
    a = p.parse_args()
    {"predict": stage_predict, "score": stage_score, "submit": stage_submit}[a.stage]()


if __name__ == "__main__":
    main()
