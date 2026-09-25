"""Ticket 17: the stack S (the average of v2 and the soft Survival Race, tuned E3 on its out-of-fold average)
and the batch expected-macro-F1 decision D, with the pre-registered upload rule.

The same code path as the hail mary's configuration A (`experiments/analysis/hailmary/hailmary.py`): the
same labelled Clients, targets, folds and seeds, with the hard race replaced by the soft one,
`SurvivalModel(order="monthly-slot", soft=True)` (what `rf train --race-order monthly-slot --soft-race`
builds). v2's out-of-fold and target probabilities are the hail mary's cached ones
(`artifacts/hailmary/<domain>/{oof,target}_v2.csv`); the soft race's are cached under
`artifacts/stack17/<domain>/`.

Domains, as in the hail mary: `rehearsal` is fitted on train and predicts the 700 selection Clients (never a
sealed-holdout Client's transactions); `final` is fitted on train plus selection and predicts the 1,000 test
Clients.

Stages:

    oof <domain>     5-fold out-of-fold probabilities of the soft race (`rf cv` folds)
    fit <domain>     fit the soft race on the labelled Clients, predict the target Clients
    gate             Gate A (train labels only): D picked per held-out fold vs nested E3 on the same folds
    predict          rehearsal: write every selection prediction to predictions/ (no labels)
    final            final: S and S+D on test (no test labels exist), written under artifacts/stack17/final/
    score            read the selection labels (once, in `data.scoring()`), score, apply the upload rule
    submit           write submissions/day2_final_<name>.csv if the rule chose a new candidate

    uv run python experiments/analysis/stack17/stack17.py <stage> ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recurring_family import data
from recurring_family import decision as decision_layer
from recurring_family.config import LABEL_COLUMN, LABELS, NONE_LABEL
from recurring_family.cross_validation import out_of_fold_proba
from recurring_family import evaluation as ev
from recurring_family.evaluation import paired_bootstrap, score
from recurring_family.submission import PREDICTION_COLUMN, write_submission
from recurring_family.survival import SurvivalModel

_spec = importlib.util.spec_from_file_location(
    "hailmary", Path(__file__).resolve().parent.parent / "hailmary" / "hailmary.py"
)
hm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hm)

PATHS = hm.PATHS
OUT = Path("experiments") / "analysis" / "stack17"
DOMAINS = hm.DOMAINS
SOFT_SELECTION_RUN = "20260925T102231-5456a7"  # the soft race's selection run, 0.5987


def cache(domain: str) -> Path:
    path = PATHS.artifacts / "stack17" / domain
    path.mkdir(parents=True, exist_ok=True)
    return path


def soft_factory():
    """The soft race exactly as `rf train --model survival --race-order monthly-slot --soft-race` builds it."""
    return lambda: SurvivalModel(order="monthly-slot", soft=True)


# --- fits ---------------------------------------------------------------------------------------------


def stage_oof(domain: str) -> None:
    make = soft_factory()
    tx, labels = hm.labelled(domain)
    with data.training_run(with_selection=domain == "final"):
        oof = out_of_fold_proba(make, tx, labels)
    oof.to_csv(cache(domain) / "oof_soft.csv")
    print(f"oof {domain} soft: {len(oof)} Clients", flush=True)


def stage_fit(domain: str) -> None:
    make = soft_factory()
    tx, labels = hm.labelled(domain)
    with data.training_run(with_selection=domain == "final"):
        fitted = make().fit(tx, labels)
    target_tx, clients = hm.target(domain)
    with data.predicting():
        proba = fitted.predict_proba(target_tx, clients)[list(LABELS)]
    proba.to_csv(cache(domain) / "target_soft.csv", index_label="client_id")
    print(f"fit {domain} soft: predicted {len(proba)} target Clients", flush=True)


# --- probabilities ------------------------------------------------------------------------------------


def read_oof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"client_id": str}).set_index("client_id")
    return df[["fold", *LABELS]]


def stack_oof(domain: str) -> pd.DataFrame:
    """S's out-of-fold probabilities: the average of v2's (the hail mary's cache) and the soft race's, on the
    same `rf cv` folds."""
    v2 = read_oof(hm.cache(domain) / "oof_v2.csv")
    soft = read_oof(cache(domain) / "oof_soft.csv")
    assert v2.index.equals(soft.index), "v2 and the soft race have different out-of-fold Clients"
    assert (v2["fold"] == soft["fold"]).all(), "v2 and the soft race have different folds"
    out = (v2[list(LABELS)] + soft[list(LABELS)]) / 2
    out.insert(0, "fold", v2["fold"])
    return out


def single_target(domain: str, model: str) -> pd.DataFrame:
    path = hm.cache(domain) / "target_v2.csv" if model == "v2" else cache(domain) / "target_soft.csv"
    return hm.read_proba(path)


def single_oof(domain: str, model: str) -> pd.DataFrame:
    path = hm.cache(domain) / "oof_v2.csv" if model == "v2" else cache(domain) / "oof_soft.csv"
    return read_oof(path)[list(LABELS)]


def stack_target(domain: str) -> pd.DataFrame:
    v2 = single_target(domain, "v2")
    return (v2 + single_target(domain, "soft").reindex(v2.index)) / 2


def e3(domain: str, oof: pd.DataFrame) -> decision_layer.Decision:
    """Tuned E3 on out-of-fold probabilities; reads the domain's labelled Clients' labels only."""
    return decision_layer.fit(oof[list(LABELS)], hm._labels(domain).reindex(oof.index))[0]


def decision_json(d: decision_layer.Decision) -> dict:
    return {"weights": d.weights, "none_threshold": d.none_threshold}


def reliability(p_none: pd.Series, is_none: pd.Series) -> list[dict]:
    """P(none) by decile of P(none): mean predicted against the observed `none` rate."""
    bins = pd.qcut(p_none.rank(method="first"), 10, labels=False)
    rows = []
    for b in range(10):
        m = bins == b
        rows.append({
            "decile": b + 1, "n": int(m.sum()), "p_low": round(float(p_none[m].min()), 4),
            "p_high": round(float(p_none[m].max()), 4), "mean_p_none": round(float(p_none[m].mean()), 4),
            "observed_none": round(float(is_none[m].mean()), 4),
        })
    return rows


def update_results(key: str, value) -> None:
    path = OUT / "results.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    result[key] = value
    path.write_text(json.dumps(result, indent=1))


def gate_passed() -> bool:
    path = OUT / "results.json"
    if not path.exists() or "gate_a" not in json.loads(path.read_text()):
        raise SystemExit("run `gate` first: D is built only if Gate A passes")
    return bool(json.loads(path.read_text())["gate_a"]["passed"])


# --- Gate A (train labels only) -----------------------------------------------------------------------

GATE_MARGIN = 0.01
EPS = 1e-9


def stage_gate() -> None:
    domain = "rehearsal"  # train only
    oof = stack_oof(domain)
    labels = hm._labels(domain).reindex(oof.index)
    assert labels.notna().all()
    nested = pd.Series(index=oof.index, dtype=object)
    batch = pd.Series(index=oof.index, dtype=object)
    folds = []
    for k in sorted(oof["fold"].unique()):
        held = oof[oof["fold"] == k][list(LABELS)]
        rest = oof[oof["fold"] != k][list(LABELS)]
        e3_k = decision_layer.fit(rest, labels.reindex(rest.index))[0]
        d_k, info = decision_layer.fit_expected(held)
        nested.loc[held.index] = e3_k.apply(held)
        batch.loc[held.index] = d_k.apply(held)
        folds.append({
            "fold": int(k), "n": len(held),
            "nested_e3": round(score(labels[held.index], nested[held.index])["macro_f1"], 4),
            "d": round(score(labels[held.index], batch[held.index])["macro_f1"], 4),
            "d_expected": round(info["expected_macro_f1"], 4),
            "e3_decision": decision_json(e3_k), "d_decision": decision_json(d_k),
        })
        print(f"gate fold {k}: nested E3 {folds[-1]['nested_e3']:.4f}  D {folds[-1]['d']:.4f} "
              f"(expected {folds[-1]['d_expected']:.4f})", flush=True)
    e3_f1 = score(labels, nested)["macro_f1"]
    d_f1 = score(labels, batch)["macro_f1"]
    passed = d_f1 >= e3_f1 - GATE_MARGIN - EPS
    gate = {
        "rule": "pass if pooled realised D >= pooled nested E3 - 0.01 (train out-of-fold, S's probabilities)",
        "nested_e3_macro_f1": round(e3_f1, 4), "d_macro_f1": round(d_f1, 4), "d_minus_e3": round(d_f1 - e3_f1, 4),
        "passed": bool(passed), "folds": folds,
        "p_none_reliability": reliability(oof[NONE_LABEL], (labels == NONE_LABEL).astype(float)),
        "none_share": {"true": round(float((labels == NONE_LABEL).mean()), 4),
                       "nested_e3": round(float((nested == NONE_LABEL).mean()), 4),
                       "d": round(float((batch == NONE_LABEL).mean()), 4)},
    }
    update_results("gate_a", gate)
    print(json.dumps({k: v for k, v in gate.items() if k not in ("folds", "p_none_reliability")}, indent=1))


# --- rehearsal predictions (no labels) ----------------------------------------------------------------

PRED = OUT / "predictions"


def write_prediction(name: str, predicted: pd.Series) -> None:
    PRED.mkdir(parents=True, exist_ok=True)
    assert predicted.notna().all() and len(predicted) == 700, f"{name}: incomplete predictions"
    predicted.rename("predicted").sort_index().to_csv(PRED / f"{name}.csv", index_label="client_id")


def stage_predict() -> None:
    domain = "rehearsal"
    passed = gate_passed()
    proba = stack_target(domain)
    decision = e3(domain, stack_oof(domain))
    out = {"S": decision.apply(proba)}
    info = {"S_e3": decision_json(decision)}
    if passed:
        d, d_info = decision_layer.fit_expected(proba)
        out["S_D"] = d.apply(proba)
        info["S_D"] = {**decision_json(d), **{k: round(v, 4) for k, v in d_info.items()}}
    for m in ("v2", "soft"):
        out[f"{m}_alone"] = e3(domain, single_oof(domain, m)).apply(single_target(domain, m))
    hm_decision, _ = hm.decision_and_prior(domain)
    out["hailmary_A"] = hm_decision.apply(hm.averaged(domain))
    for name, predicted in out.items():
        write_prediction(name, predicted)
    proba.to_csv(cache(domain) / "S_proba.csv", index_label="client_id")
    update_results("rehearsal_decisions", info)
    print(f"predict: wrote {sorted(out)} to {PRED}")


# --- final predictions on test (no test labels exist) -------------------------------------------------

NEW = {"S": "day2_final_stack", "S_D": "day2_final_stack_d"}
EXISTING = {
    "v2": "day2_final_ranker_none_v2", "soft race": "day2_final_survival_soft", "hail mary A": "day2_final_hailmary",
}


def stage_final() -> None:
    domain = "final"
    proba = stack_target(domain)
    decision = e3(domain, stack_oof(domain))
    out = {"S": decision.apply(proba)}
    info = {"S_e3": decision_json(decision)}
    if gate_passed():
        d, d_info = decision_layer.fit_expected(proba)
        out["S_D"] = d.apply(proba)
        info["S_D"] = {**decision_json(d), **{k: round(v, 4) for k, v in d_info.items()}}
    # checks: each single model's refit reproduces its committed test file
    checks = {}
    for m, name in (("v2", EXISTING["v2"]), ("soft", EXISTING["soft race"])):
        mine = e3(domain, single_oof(domain, m)).apply(single_target(domain, m))
        committed = committed_test(name)
        checks[f"{m} alone vs {name}.csv"] = round(float((mine.reindex(committed.index) == committed).mean()), 4)
        assert checks[f"{m} alone vs {name}.csv"] == 1.0, f"{m}: final refit does not reproduce {name}.csv"
    expected = data.load_sample_submission(PATHS.raw)["client_id"].astype(str)
    for name, predicted in out.items():
        assert predicted.notna().all() and set(predicted.index) == set(expected), f"{name}: incomplete test predictions"
        predicted.rename("predicted").sort_index().to_csv(cache(domain) / f"{name}_test.csv", index_label="client_id")
    proba.to_csv(cache(domain) / "S_proba.csv", index_label="client_id")
    update_results("final", {"decisions": info, "single_model_test_file_agreement": checks,
                             "label_mix": {n: p.value_counts().to_dict() for n, p in out.items()}})
    print(json.dumps(checks, indent=1))


def committed_test(name: str) -> pd.Series:
    df = pd.read_csv(PATHS.submissions / f"{name}.csv", dtype=str, keep_default_na=False)
    return df.set_index("client_id")[PREDICTION_COLUMN]


def new_test(name: str) -> pd.Series:
    path = cache("final") / f"{name}_test.csv"
    if not path.exists():
        raise SystemExit("run `final` first: the upload rule needs each new candidate's test file")
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("client_id")["predicted"]


# --- scoring and the upload rule ----------------------------------------------------------------------

SHRINK = 1.7
MIN_DISAGREEMENT = 0.10


def paired_se(truth: pd.Series, new: pd.Series, old: pd.Series) -> float:
    """The standard deviation of the paired bootstrap's deltas, over the same resamples `paired_bootstrap`
    draws (2,000, seed 0)."""
    t, n, o = ev._one_hot(truth), ev._one_hot(new), ev._one_hot(old)
    w = ev._resample_weights(len(t))
    diffs = ev._weighted_macro_f1(w, t, n) - ev._weighted_macro_f1(w, t, o)
    return float(np.std(diffs, ddof=1))


def platt_slope(p: pd.Series, y: pd.Series) -> dict:
    from sklearn.linear_model import LogisticRegression

    q = p.clip(1e-6, 1 - 1e-6)
    logit = np.log(q / (1 - q)).to_numpy()[:, None]
    fit = LogisticRegression(penalty=None).fit(logit, y.astype(int).to_numpy())
    return {"slope": round(float(fit.coef_[0][0]), 4), "intercept": round(float(fit.intercept_[0]), 4)}


def read_prediction(name: str) -> pd.Series:
    path = PRED / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"run `predict` first: {path} is missing")
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("client_id")["predicted"]


def stage_score() -> None:
    passed = gate_passed()
    names = ["S", *(["S_D"] if passed else []), "v2_alone", "soft_alone", "hailmary_A"]
    predictions = {n: read_prediction(n) for n in names}
    for n, p in predictions.items():
        assert p.notna().all() and len(p) == 700, f"{n}: incomplete predictions"
    tests = {n: new_test(n) for n in NEW if n in predictions}
    # every prediction is frozen; only now read the selection labels
    with data.scoring():
        truth = data.load_labels(PATHS.raw, "selection").set_index("client_id")[LABEL_COLUMN]
    truth.index = truth.index.astype(str)
    runs = {
        name: pd.read_csv(PATHS.run_predictions(run), dtype=str, keep_default_na=False)
        .set_index("client_id")["predicted"].reindex(truth.index)
        for name, run in (("v2", hm.V2_SELECTION_RUN), ("soft", SOFT_SELECTION_RUN))
    }
    assert all(r.notna().all() for r in runs.values())
    predictions = {n: p.reindex(truth.index) for n, p in predictions.items()}
    checks = {}
    for m in ("v2", "soft"):
        same = float((predictions[f"{m}_alone"] == runs[m]).mean())
        checks[f"{m} alone vs logged run"] = same
        assert same == 1.0, f"{m} alone does not reproduce its logged run ({same:.3f} agreement)"
    committed_a = (
        pd.read_csv(hm.OUT / "predictions" / "A.csv", dtype=str, keep_default_na=False)
        .set_index("client_id")["predicted"].reindex(truth.index)
    )
    same = float((predictions["hailmary_A"] == committed_a).mean())
    checks["hail mary A vs its committed predictions"] = same
    assert same == 1.0, f"hail mary A does not reproduce its committed predictions ({same:.3f})"

    v2_test = committed_test(EXISTING["v2"])
    candidates = {
        # name: (selection predictions, test file predictions, file)
        "v2": (runs["v2"], v2_test, EXISTING["v2"]),
        "soft race": (runs["soft"], committed_test(EXISTING["soft race"]), EXISTING["soft race"]),
        "hail mary A": (predictions["hailmary_A"], committed_test(EXISTING["hail mary A"]), EXISTING["hail mary A"]),
        "S": (predictions["S"], tests["S"], NEW["S"]),
    }
    if passed:
        candidates["S+D"] = (predictions["S_D"], tests["S_D"], NEW["S_D"])
    rows = []
    for name, (sel, test, file) in candidates.items():
        s = score(truth, sel)
        cmp = paired_bootstrap(truth, sel, runs["v2"])
        se = paired_se(truth, sel, runs["v2"])
        disagree = float((test.reindex(v2_test.index) != v2_test).mean())
        rows.append({
            "candidate": name, "file": f"submissions/{file}.csv", "new": name in ("S", "S+D"),
            "macro_f1": round(s["macro_f1"], 4), "d": round(cmp["delta"], 4), "d_low": round(cmp["low"], 4),
            "d_high": round(cmp["high"], 4), "se_paired": round(se, 4), "d_shrunk": round(cmp["delta"] - SHRINK * se, 4),
            "test_disagreement_v2": round(disagree, 4), "selection_agreement_v2": round(float((sel == runs["v2"]).mean()), 4),
            "in_pool": bool(cmp["delta"] > 0 and disagree >= MIN_DISAGREEMENT - EPS),
            "_d_shrunk_raw": cmp["delta"] - SHRINK * se,
            **{f"f1_{l}": round(s[f"f1_{l}"], 4) for l in LABELS},
        })
    pool = sorted((r for r in rows if r["in_pool"]), key=lambda r: r["_d_shrunk_raw"], reverse=True)
    if pool:
        chosen = pool[0]
        rule = {"chosen": chosen["candidate"], "file": chosen["file"], "new": chosen["new"],
                "why": "rule 3: the pool's top by d - 1.7 SE_paired"}
    else:
        rule = {"chosen": None, "file": f"submissions/{EXISTING['hail mary A']}.csv", "new": False,
                "why": "rule 3: the pool is empty, so the committed hail-mary file stays"}
    rule["pool"] = [{k: r[k] for k in ("candidate", "d", "se_paired", "d_shrunk", "test_disagreement_v2", "file")}
                    for r in pool]
    for r in rows:
        r.pop("_d_shrunk_raw")
    # post hoc, explanation only: the calibration of S's P(none) on selection
    s_proba = hm.read_proba(cache("rehearsal") / "S_proba.csv").reindex(truth.index)
    is_none = (truth == NONE_LABEL).astype(float)
    calibration = {"reliability": reliability(s_proba[NONE_LABEL], is_none),
                   "platt": platt_slope(s_proba[NONE_LABEL], is_none),
                   "mean_p_none": round(float(s_proba[NONE_LABEL].mean()), 4), "none_share": round(float(is_none.mean()), 4)}
    update_results("rehearsal", {"checks": checks, "rows": rows, "s_p_none_calibration_post_hoc": calibration})
    update_results("upload_rule", rule)
    for r in rows:
        print(f"{r['candidate']:<12} {r['macro_f1']:.4f}  d {r['d']:+.4f} ({r['d_low']:+.4f} .. {r['d_high']:+.4f})"
              f"  SE {r['se_paired']:.4f}  d~ {r['d_shrunk']:+.4f}  test disagree {r['test_disagreement_v2']:.3f}"
              f"  pool {r['in_pool']}")
    print(json.dumps(rule, indent=1))
    print(json.dumps(calibration["platt"]))


def stage_submit() -> None:
    rule = json.loads((OUT / "results.json").read_text())["upload_rule"]
    if not rule["new"]:
        raise SystemExit(f"the rule chose an existing file ({rule['file']}); nothing to write")
    key = {"S": "S", "S+D": "S_D"}[rule["chosen"]]
    predicted = new_test(key)
    expected = data.load_sample_submission(PATHS.raw)["client_id"]
    out = PATHS.submissions / f"{NEW[key]}.csv"
    write_submission(predicted, expected, out)
    print(f"submit: {rule['chosen']} -> {out} ({len(expected)} Clients); label mix:")
    print(predicted.value_counts().to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="stage", required=True)
    for stage in ("oof", "fit"):
        s = sub.add_parser(stage)
        s.add_argument("domain", choices=DOMAINS)
    for stage in ("gate", "predict", "final", "score", "submit"):
        sub.add_parser(stage)
    a = p.parse_args()
    if a.stage in ("oof", "fit"):
        {"oof": stage_oof, "fit": stage_fit}[a.stage](a.domain)
    else:
        {"gate": stage_gate, "predict": stage_predict, "final": stage_final, "score": stage_score,
         "submit": stage_submit}[a.stage]()


if __name__ == "__main__":
    main()
