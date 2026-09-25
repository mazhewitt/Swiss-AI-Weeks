"""Ticket 18: is there a `none` signal of the target domain's own? Pre-registered, capped diagnostic.

Data: the 700 selection Clients only, inside `data.training_run(with_selection=True)`; their labels are the
selection labels. The sealed holdout is never read (neither its labels nor its transactions).
Blocks: base (the v2 `none` model's stream features) and raw (ticket 16's kitchen sink without description
words or bags, plus Decoy, currency / hour / MCC mix and refund features); see `features.py`.
Model: LightGBM with the `none` model's small fixed parameters (`n_jobs=1`, no tuning), binary `none` against
family, repeated 5x5 stratified CV (seeds 0-4), out-of-fold scores averaged over the repeats.
Measures: raw AUC with a DeLong 95% CI (the gate); base and base+raw AUCs and a paired DeLong test of base+raw
against base; the label-free train-vs-test adversarial AUC of the top 10 raw features.
Gate: raw AUC >= 0.70 with the CI excluding 0.5 -> recommend a build; otherwise close the route.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "experiments/analysis/target_none")
from features import blocks  # noqa: E402
from recurring_family import data  # noqa: E402
from recurring_family.cli import Paths, _training_data  # noqa: E402
from recurring_family.none_model import NONE_PARAMS  # noqa: E402

OUT = Path("experiments/analysis/target_none")
CACHE = OUT / "cache"
PARAMS = {**NONE_PARAMS, "n_jobs": 1}
SEEDS = range(5)
GATE_AUC = 0.70


# --- DeLong (Sun & Xu 2014, fast midrank form) ------------------------------------------------


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(y: np.ndarray, scores: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """AUCs of each score vector and their DeLong covariance matrix."""
    y = np.asarray(y).astype(bool)
    pos, neg = [np.asarray(s)[y] for s in scores], [np.asarray(s)[~y] for s in scores]
    m, n = y.sum(), (~y).sum()
    aucs, v10, v01 = [], [], []
    for p, q in zip(pos, neg):
        tx, ty, tz = _midrank(p), _midrank(q), _midrank(np.concatenate([p, q]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
    v10, v01 = np.array(v10), np.array(v01)
    cov = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
    return np.array(aucs), cov


def auc_ci(y, s) -> dict:
    a, cov = delong(y, [s])
    se = float(np.sqrt(cov[0, 0]))
    z = (a[0] - 0.5) / se
    return {"auc": round(float(a[0]), 4), "ci95": [round(float(a[0] - 1.96 * se), 4), round(float(a[0] + 1.96 * se), 4)],
            "se": round(se, 4), "p_vs_0.5": float(2 * stats.norm.sf(abs(z)))}


def paired(y, s_new, s_old) -> dict:
    a, cov = delong(y, [s_new, s_old])
    diff = a[0] - a[1]
    se = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    z = diff / se
    return {"auc_new": round(float(a[0]), 4), "auc_old": round(float(a[1]), 4), "diff": round(float(diff), 4),
            "ci95": [round(float(diff - 1.96 * se), 4), round(float(diff + 1.96 * se), 4)], "z": round(float(z), 3),
            "p_two_sided": float(2 * stats.norm.sf(abs(z)))}


# --- models -----------------------------------------------------------------------------------


def repeated_oof(x: pd.DataFrame, y: pd.Series, name: str) -> tuple[pd.Series, pd.Series, list[float]]:
    """Out-of-fold scores averaged over 5 repeats of 5-fold stratified CV (seeds 0-4); summed gain."""
    total = pd.Series(0.0, index=y.index)
    gain = pd.Series(0.0, index=x.columns)
    per_repeat = []
    for seed in SEEDS:
        p = pd.Series(np.nan, index=y.index)
        for fit, score in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
            m = lgb.LGBMClassifier(**PARAMS).fit(x.iloc[fit], y.iloc[fit])
            p.iloc[score] = m.predict_proba(x.iloc[score])[:, 1]
            gain += pd.Series(m.booster_.feature_importance("gain"), index=x.columns)
        per_repeat.append(round(roc_auc_score(y, p), 4))
        total += p
    avg = total / len(SEEDS)
    print(f"{name:10s} none AUC {roc_auc_score(y, avg):.4f} (repeats {per_repeat}; {x.shape[1]} features)", flush=True)
    return avg, gain.sort_values(ascending=False), per_repeat


def adversarial(a: pd.DataFrame, b: pd.DataFrame, name: str) -> float:
    """Label-free domain classifier: 5-fold out-of-fold AUC of `b` rows against `a` rows."""
    x = pd.concat([a, b], ignore_index=True)
    y = pd.Series([0] * len(a) + [1] * len(b))
    p = pd.Series(np.nan, index=y.index)
    for fit, score in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
        m = lgb.LGBMClassifier(**PARAMS).fit(x.iloc[fit], y.iloc[fit])
        p.iloc[score] = m.predict_proba(x.iloc[score])[:, 1]
    auc = roc_auc_score(y, p)
    print(f"adversarial {name:28s} AUC {auc:.4f}", flush=True)
    return float(auc)


def main():
    paths = Paths(Path("."))
    results = {"design": {"clients": "700 selection Clients", "cv": "5 repeats x 5-fold stratified, seeds 0-4, oof averaged",
                          "params": {k: v for k, v in PARAMS.items() if k != "verbose"}, "gate": f"raw AUC >= {GATE_AUC} and CI excludes 0.5"}}
    with data.training_run(with_selection=True):
        tx, labels = _training_data(paths, with_selection=True)
        train_ids = set(data.load_labels(paths.raw, "train")["client_id"].astype(str))
        selection = pd.Index(sorted(c for c in labels.index.astype(str) if c not in train_ids), name="client_id")
        y = (labels.reindex(selection) == "none").astype(int)
        sel_tx = tx[tx["client_id"].astype(str).isin(set(selection))].reset_index(drop=True)
    # from here on no label is read: y is fixed, features come from transactions only
    assert len(selection) == 700, len(selection)
    results["n"] = {"clients": int(len(y)), "none": int(y.sum()), "family": int((1 - y).sum())}
    print(results["n"], flush=True)
    base, raw, mccs = blocks(sel_tx, selection, cache=CACHE / "selection")
    results["n_features"] = {"base": base.shape[1], "raw": raw.shape[1]}
    results["base_clients_without_streams"] = int(base.isna().all(axis=1).sum())

    scores = {}
    for name, x in [("raw", raw), ("base", base), ("base+raw", pd.concat([base, raw], axis=1))]:
        s, gain, reps = repeated_oof(x, y, name)
        scores[name] = s
        results[name] = {**auc_ci(y.to_numpy(), s.to_numpy()), "per_repeat_auc": reps,
                         "top_gain": gain.head(15).round(0).to_dict()}
    results["paired_delong_base+raw_vs_base"] = paired(y.to_numpy(), scores["base+raw"].to_numpy(), scores["base"].to_numpy())
    pd.DataFrame(scores).assign(none=y).to_csv(CACHE / "oof_scores.csv")

    # single-feature AUCs of the top 10 raw features (direction-free), for reading
    top10 = list(results["raw"]["top_gain"])[:10]
    single = {}
    for c in top10:
        v = raw[c].fillna(raw[c].median())
        a = roc_auc_score(y, v)
        single[c] = {"auc": round(max(a, 1 - a), 4), "direction": "high -> none" if a >= 0.5 else "low -> none",
                     "mean_none": round(float(v[y == 1].mean()), 3), "mean_family": round(float(v[y == 0].mean()), 3)}
    results["raw_top10_single_feature"] = single

    # label-free adversarial AUC of the top 10 raw features: train split vs test, and selection vs test
    domains = {}
    for split in ("train", "test"):
        t = data.load_transactions(paths.raw, split)
        clients = pd.Index(sorted(t["client_id"].astype(str).unique()), name="client_id")
        t = t.assign(client_id=t["client_id"].astype(str))
        _, domains[split], _ = blocks(t, clients, mccs=mccs, cache=CACHE / split)
    sel_raw = raw.copy()
    results["adversarial_top10_raw"] = {
        "features": top10,
        "train_vs_test": round(adversarial(domains["train"][top10], domains["test"][top10], "train vs test"), 4),
        "selection_vs_test": round(adversarial(sel_raw[top10], domains["test"][top10], "selection vs test"), 4),
        "train_vs_selection": round(adversarial(domains["train"][top10], sel_raw[top10], "train vs selection"), 4),
    }

    raw_r = results["raw"]
    passed = raw_r["auc"] >= GATE_AUC and raw_r["ci95"][0] > 0.5
    results["gate"] = {"raw_auc": raw_r["auc"], "raw_ci95": raw_r["ci95"], "passed": bool(passed),
                       "verdict": "recommend a build (Daume / is_target stack) as a new ticket" if passed
                       else "close the target-domain none route"}
    print(json.dumps({k: results[k] for k in ("raw", "base", "base+raw")}, indent=1, default=str))
    print(json.dumps(results["paired_delong_base+raw_vs_base"]), json.dumps(results["adversarial_top10_raw"]), json.dumps(results["gate"]))
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
