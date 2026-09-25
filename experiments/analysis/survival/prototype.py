"""Ticket 13 prototype: the Survival Race on train out-of-fold (folds of `rf cv`). Reads no valid labels.

Each Candidate Stream gets a survival probability s; streams are ordered by projected next payment
(no projection last, ties by more payments first). P(stream i next) = s_i * prod_{j<i}(1 - s_j),
P(none) = prod_j (1 - s_j). Training rows: every stream up to and including the label's first stream
(target 1 on it, 0 before it); later streams are censored; `none` Clients give all streams target 0;
Clients whose label family has no candidate stream give no rows.
"""
import json, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "scripts")
from none_model_variants import macro_f1s  # noqa: E402
from recurring_family import data
from recurring_family.cli import Paths, _training_data
from recurring_family.config import CUTOFF, LABELS, NONE_LABEL
from recurring_family.cross_validation import CV_FOLDS, CV_SEED
from recurring_family.none_model import CLIENT_CHURN, CHURN_FIELDS, _stream_fields, client_features
from recurring_family.ranker import FEATURE_COLUMNS as RANKER_FEATURES, LGBM_PARAMS, _detected, candidates

OUT = Path("experiments/analysis/survival")
FAMILIES = [l for l in LABELS if l != NONE_LABEL]


def stream_table(transactions, clients):
    streams, payments = _detected(transactions, clients)
    streams = streams.reset_index(drop=True)
    cand = candidates(streams)
    sf = _stream_fields(streams, payments, CUTOFF)
    churn_cols = [c for c in CHURN_FIELDS if c in sf.columns and c not in cand.columns]
    t = pd.concat([cand.reset_index(drop=True), sf[churn_cols].reset_index(drop=True)], axis=1)
    cc = client_features(streams, payments)[list(CLIENT_CHURN)]
    t = t.join(cc, on="client_id")
    t["_key"] = t["days_to_next"].fillna(np.inf)
    t = t.assign(_n=-t["n_payments"]).sort_values(["client_id", "_key", "_n"], kind="stable")
    t["order"] = t.groupby("client_id").cumcount()
    return t.drop(columns=["_key", "_n"]).reset_index(drop=True), churn_cols


def training_rows(t, labels):
    lab = t["client_id"].map(labels.astype(str))
    fam = t["family"].astype(str)
    hit = (fam == lab)
    first_hit = hit & (hit.astype(int).groupby(t["client_id"]).cumsum() == 1)
    pos = t["order"].where(first_hit).groupby(t["client_id"]).transform("min")
    is_none = lab == NONE_LABEL
    explained = is_none | pos.notna()
    keep = explained & (is_none | (t["order"] <= pos))
    target = (first_hit & keep).astype(int)
    return keep.to_numpy(), target.to_numpy()


def combine(t, s, clients):
    t = t.assign(s=s)
    t["log_stop_before"] = t.groupby("client_id")["s"].transform(lambda x: np.concatenate([[0.0], np.cumsum(np.log1p(-np.clip(x.to_numpy()[:-1], 0, 1 - 1e-9)))]))
    t["p"] = t["s"] * np.exp(t["log_stop_before"])
    fam = t.pivot_table(index="client_id", columns=t["family"].astype(str), values="p", aggfunc="sum", observed=True)
    fam = fam.reindex(index=clients, columns=FAMILIES).fillna(0.0)
    none = 1 - fam.sum(axis=1)
    out = fam.assign(**{NONE_LABEL: none.clip(lower=0)})
    return out[list(LABELS)]


def run(t, labels, features, folds):
    oof, rows = [], []
    for k, (fit_c, score_c) in enumerate(folds):
        ft, st_ = t[t.client_id.isin(fit_c)], t[t.client_id.isin(score_c)]
        keep, y = training_rows(ft, labels)
        model = lgb.LGBMClassifier(**LGBM_PARAMS).fit(ft.loc[keep, features], y[keep])
        s = model.predict_proba(st_[features])[:, 1]
        oof.append(combine(st_, s, pd.Index(score_c)).assign(fold=k))
        k2, y2 = training_rows(st_, labels)
        rows.append(pd.DataFrame({"s": s[k2], "y": y2[k2], "order": st_["order"].to_numpy()[k2]}))
    oof = pd.concat(oof).reindex(labels.index)
    return oof, pd.concat(rows)


def calibration(t, oof, labels):
    act = t[t.active == 1].groupby("client_id")["family"].nunique().reindex(labels.index).fillna(0).astype(int)
    r = []
    for k in (1, 2, 3, 4):
        ids = act[act == k].index
        r.append({"k_active_families": k, "n": len(ids), "pred_none": round(oof.loc[ids, NONE_LABEL].mean(), 3),
                  "obs_none": round((labels[ids] == NONE_LABEL).mean(), 3)})
    return r


def main():
    paths = Paths(Path("."))
    with data.training_run():
        transactions, labels = _training_data(paths, with_selection=False)
    labels = pd.Series(labels.astype(str).to_numpy(), index=labels.index.astype(str))
    t, churn_cols = stream_table(transactions, labels.index)
    keep, _ = training_rows(t, labels)
    unexplained = int(((labels != NONE_LABEL) & ~labels.index.isin(t.loc[t["family"].astype(str) == t["client_id"].map(labels), "client_id"])).sum())
    print(f"{len(t)} candidate streams; {keep.sum()} training rows; {unexplained} unexplained Clients dropped")
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    folds = [(labels.index[a], labels.index[b]) for a, b in splitter.split(labels.index, labels.to_numpy())]
    variants = {
        "S-rank: ranker features": list(RANKER_FEATURES),
        "S-full: ranker + churn + client churn": [*RANKER_FEATURES, *churn_cols, *CLIENT_CHURN],
    }
    results = []
    for name, feats in variants.items():
        oof, rows = run(t, labels, feats, folds)
        oof.to_csv(OUT / f"oof_{name.split(':')[0]}.csv", index_label="client_id")
        res = {"variant": name, **macro_f1s(oof, labels), "s_auc": round(roc_auc_score(rows.y, rows.s), 4),
               "s_auc_first_stream": round(roc_auc_score(rows[rows.order == 0].y, rows[rows.order == 0].s), 4),
               "mean_s": round(rows.s.mean(), 3), "calibration": calibration(t, oof, labels)}
        print(json.dumps(res), flush=True)
        results.append(res)
    (OUT / "prototype.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
