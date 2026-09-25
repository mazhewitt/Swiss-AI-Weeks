"""Ticket 20: do sociological churn mechanisms predict `none`? Pre-registered diagnostic, no build.

Features: the six in `features.py`, fixed and committed before this script (or anything else) read a label.
Base: the logit of hail mary's train-only P(none), the average of its v2 and Survival Race members:
  train     -> 5-fold out-of-fold, artifacts/hailmary/rehearsal/oof_{v2,surv}.csv
  selection -> artifacts/hailmary/rehearsal/target_{v2,surv}.csv (fitted on train only)
Model: logistic regression on standardised inputs, L2, C = 1; base only against base + 6.
Samples: train (2,000 Clients, train labels) and selection (700 Clients, labels read only inside
`data.training_run(with_selection=True)`); each with repeated 5x5 stratified CV (seeds 0-4), out-of-fold
scores averaged over the repeats. The sealed holdout is never read.
Measures per sample: base and base + 6 `none` AUCs, a paired DeLong test of the difference, each feature's
standalone AUC, and the coefficients with their signs against the sociological prediction.
Gate: difference >= +0.01 and its paired DeLong 95% CI excludes 0, on train and on selection -> recommend a build.
Label-free: each feature's train-vs-test adversarial AUC.

    OMP_NUM_THREADS=3 uv run python experiments/analysis/sociology/diagnostic.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "experiments/analysis/sociology")
from features import FEATURES, PREDICTED_SIGN, sociology_features  # noqa: E402
from recurring_family import data  # noqa: E402
from recurring_family.cli import Paths, _training_data  # noqa: E402

OUT = Path("experiments/analysis/sociology")
CACHE = OUT / "cache"
REHEARSAL = Path("artifacts/hailmary/rehearsal")
SEEDS = range(5)
GATE_DIFF = 0.01
EPS = 1e-6


# --- DeLong (Sun & Xu 2014, fast midrank form), as in experiments/analysis/target_none/diagnostic.py -----


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
    return {"auc": round(float(a[0]), 4), "ci95": [round(float(a[0] - 1.96 * se), 4), round(float(a[0] + 1.96 * se), 4)],
            "p_vs_0.5": float(2 * stats.norm.sf(abs((a[0] - 0.5) / se)))}


def paired(y, s_new, s_old) -> dict:
    a, cov = delong(y, [s_new, s_old])
    diff = a[0] - a[1]
    se = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    z = diff / se
    return {"auc_new": round(float(a[0]), 4), "auc_old": round(float(a[1]), 4), "diff": round(float(diff), 4),
            "ci95": [round(float(diff - 1.96 * se), 4), round(float(diff + 1.96 * se), 4)], "z": round(float(z), 3),
            "p_two_sided": float(2 * stats.norm.sf(abs(z)))}


# --- model ------------------------------------------------------------------------------------------


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))  # L2 is the default


def repeated_oof(x: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, list[float], pd.DataFrame]:
    """Out-of-fold scores averaged over 5 repeats of 5-fold stratified CV (seeds 0-4); per-fold coefficients."""
    total = pd.Series(0.0, index=y.index)
    per_repeat, coefs = [], []
    for seed in SEEDS:
        p = pd.Series(np.nan, index=y.index)
        for fit, score in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
            m = _lr().fit(x.iloc[fit], y.iloc[fit])
            p.iloc[score] = m.predict_proba(x.iloc[score])[:, 1]
            coefs.append(m[-1].coef_[0])
        per_repeat.append(round(roc_auc_score(y, p), 4))
        total += p
    return total / len(SEEDS), per_repeat, pd.DataFrame(coefs, columns=x.columns)


def base_logit(files: list[Path], clients: pd.Index) -> pd.Series:
    p = sum(pd.read_csv(f, dtype={"client_id": str}).set_index("client_id")["none"] for f in files) / len(files)
    p = p.reindex(clients)
    assert p.notna().all(), "base P(none) missing for some Clients"
    p = p.clip(EPS, 1 - EPS)
    return np.log(p / (1 - p)).rename("base")


def sample_test(name: str, feats: pd.DataFrame, base: pd.Series, y: pd.Series) -> dict:
    x0 = base.to_frame()
    x6 = pd.concat([base, feats], axis=1)
    s0, r0, c0 = repeated_oof(x0, y)
    s6, r6, c6 = repeated_oof(x6, y)
    yy = y.to_numpy()
    full = _lr().fit(x6, y)[-1].coef_[0]
    coefs = {}
    for i, c in enumerate(x6.columns):
        pred = PREDICTED_SIGN.get(c)
        coefs[c] = {"coef_full_fit": round(float(full[i]), 4),
                    "coef_fold_mean": round(float(c6[c].mean()), 4),
                    "fold_share_positive": round(float((c6[c] > 0).mean()), 2),
                    **({"predicted_sign": "+" if pred > 0 else "-",
                        "matches_prediction": bool(np.sign(full[i]) == pred)} if pred else {})}
    single = {}
    for c in FEATURES:
        a = auc_ci(yy, feats[c].to_numpy())
        pred = PREDICTED_SIGN[c]
        single[c] = {"auc_high_to_none": a["auc"], "ci95": a["ci95"],
                     "auc_predicted_direction": round(a["auc"] if pred > 0 else 1 - a["auc"], 4),
                     "mean_none": round(float(feats.loc[y == 1, c].mean()), 4),
                     "mean_family": round(float(feats.loc[y == 0, c].mean()), 4)}
    res = {"n": {"clients": int(len(y)), "none": int(y.sum())},
           "base_raw_auc": auc_ci(yy, base.to_numpy()),
           "base_lr": {**auc_ci(yy, s0.to_numpy()), "per_repeat_auc": r0},
           "base_plus_6_lr": {**auc_ci(yy, s6.to_numpy()), "per_repeat_auc": r6},
           "paired_delong_base_plus_6_vs_base": paired(yy, s6.to_numpy(), s0.to_numpy()),
           "coefficients_standardised": coefs,
           "standalone_auc": single}
    d = res["paired_delong_base_plus_6_vs_base"]
    print(f"{name}: base {d['auc_old']} base+6 {d['auc_new']} diff {d['diff']} CI {d['ci95']}", flush=True)
    return res


def adversarial(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Label-free: each feature's direction-free AUC separating `b` rows from `a` rows, and the six together
    (logistic regression, 5-fold out-of-fold)."""
    x = pd.concat([a, b], ignore_index=True)
    y = pd.Series([0] * len(a) + [1] * len(b))
    out = {}
    for c in FEATURES:
        v = roc_auc_score(y, x[c])
        out[c] = round(float(max(v, 1 - v)), 4)
    p = pd.Series(np.nan, index=y.index)
    for fit, score in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
        p.iloc[score] = _lr().fit(x.iloc[fit], y.iloc[fit]).predict_proba(x.iloc[score])[:, 1]
    out["all_six_lr_oof"] = round(float(roc_auc_score(y, p)), 4)
    return out


def main():
    paths = Paths(Path("."))
    CACHE.mkdir(parents=True, exist_ok=True)
    results = {"design": {"model": "standardised logistic regression, L2, C=1; base only vs base + 6",
                          "cv": "5 repeats x 5-fold stratified, seeds 0-4, oof averaged",
                          "base": "logit of the mean of hail mary v2 and surv train-only P(none)",
                          "gate": f"diff >= {GATE_DIFF} and paired DeLong 95% CI excludes 0, on train and selection"}}

    # label-free features: train and test
    feats = {}
    for split in ("train", "test"):
        t = data.load_transactions(paths.raw, split)
        clients = pd.Index(sorted(t["client_id"].astype(str).unique()), name="client_id")
        feats[split] = sociology_features(t, clients)
        feats[split].to_pickle(CACHE / f"features_{split}.pkl")

    # selection: labels and transactions only inside the training run that allows them
    with data.training_run(with_selection=True):
        tx, labels = _training_data(paths, with_selection=True)
        labels.index = labels.index.astype(str)
        train_labels = data.load_labels(paths.raw, "train").set_index("client_id")["target_next_recurring_merchant"]
        train_labels.index = train_labels.index.astype(str)
        selection = pd.Index(sorted(c for c in labels.index if c not in set(train_labels.index)), name="client_id")
        y_sel = (labels.reindex(selection) == "none").astype(int)
        sel_tx = tx[tx["client_id"].astype(str).isin(set(selection))].reset_index(drop=True)
    assert len(selection) == 700, len(selection)
    feats["selection"] = sociology_features(sel_tx, selection)
    feats["selection"].to_pickle(CACHE / "features_selection.pkl")

    train_ids = pd.Index(sorted(train_labels.index), name="client_id")
    y_tr = (train_labels.reindex(train_ids) == "none").astype(int)
    assert len(train_ids) == 2000
    base_tr = base_logit([REHEARSAL / "oof_v2.csv", REHEARSAL / "oof_surv.csv"], train_ids)
    base_sel = base_logit([REHEARSAL / "target_v2.csv", REHEARSAL / "target_surv.csv"], selection)

    results["train"] = sample_test("train", feats["train"].reindex(train_ids), base_tr, y_tr)
    results["selection"] = sample_test("selection", feats["selection"].reindex(selection), base_sel, y_sel)

    results["adversarial_auc"] = {
        "train_vs_test": adversarial(feats["train"], feats["test"]),
        "selection_vs_test": adversarial(feats["selection"], feats["test"]),
        "train_vs_selection": adversarial(feats["train"], feats["selection"]),
    }
    print(json.dumps(results["adversarial_auc"]), flush=True)

    checks = {}
    for s in ("train", "selection"):
        d = results[s]["paired_delong_base_plus_6_vs_base"]
        checks[s] = {"diff": d["diff"], "ci95": d["ci95"], "diff_ok": d["diff"] >= GATE_DIFF, "ci_excludes_0": d["ci95"][0] > 0}
    passed = all(c["diff_ok"] and c["ci_excludes_0"] for c in checks.values())
    results["gate"] = {**checks, "passed": bool(passed),
                       "verdict": "recommend a build as a new ticket" if passed else "close the sociological route"}
    print(json.dumps(results["gate"]), flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
