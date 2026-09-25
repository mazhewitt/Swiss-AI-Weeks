"""Filler Description features for the Survival Race (soft race, monthly-slot), train 5-fold out-of-fold.
Variants: base (the 16 ranker features), R (relative filler features: within-Client differences and ranks,
which a base-rate shift cannot move), RA (R plus absolute per-stream filler counts), RAC (RA plus Client-level
filler counts). For each: nested tuned macro-F1, s AUC, and the extra features' train-vs-test drift AUC."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "scripts")
from none_model_variants import macro_f1s
from recurring_family import data
from recurring_family.cli import Paths, _training_data
from recurring_family.config import CUTOFF, LABELS
from recurring_family.cross_validation import CV_FOLDS, CV_SEED
from recurring_family.ranker import FEATURE_COLUMNS, LGBM_PARAMS, _detected, candidates
from recurring_family.streams import _decoy_described, _names_family
from recurring_family.survival import jitter_scale, race_order, race_slot, soft_race_proba, training_rows

OUT = Path("experiments/analysis/none_probe")
DAY = pd.Timedelta(days=1)
STREAM_ABS = ["fl_n", "fl_share", "fl_last_days", "fl_n_90"]
CLIENT_ABS = ["fl_client_share", "fl_client_n", "fl_client_stray_n", "fl_client_stray_90"]
RELATIVE = ["fl_share_minus_client", "fl_share_rank", "fl_share_of_client_fillers", "fl_last_days_minus_client_min"]
VARIANTS = {"base": [], "R": RELATIVE, "RA": RELATIVE + STREAM_ABS, "RAC": RELATIVE + STREAM_ABS + CLIENT_ABS}


def filler_table(tx: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
    """The soft race table of `clients` (race order monthly-slot) with the filler features attached."""
    streams, paid = _detected(tx, clients)
    card = tx[(tx["type"] == "card_payment") & tx["client_id"].astype(str).isin(set(map(str, clients)))].copy()
    card["client_id"] = card["client_id"].astype(str)
    card["generic"] = ~card["description"].map(_names_family) & ~card["description"].map(_decoy_described)
    card["days"] = (CUTOFF - card["timestamp"]) / DAY
    card["key"] = card["client_id"] + "|" + card["timestamp"].astype(str) + "|" + card["amount"].round(2).astype(str)
    p = paid.assign(client_id=paid["client_id"].astype(str))
    p["key"] = p["client_id"] + "|" + p["timestamp"].astype(str) + "|" + p["amount"].round(2).astype(str)
    p = p.merge(card[["key", "generic", "days"]].drop_duplicates("key"), on="key", how="left")
    p["generic"] = p["generic"].fillna(False).astype(bool)
    g = p.groupby(["client_id", "stream_id"])
    st = pd.DataFrame({"fl_n": g["generic"].sum(), "n_pay": g.size(), "fl_last_days": g.apply(lambda d: d.loc[d["generic"], "days"].min()),
                       "fl_n_90": g.apply(lambda d: (d["generic"] & (d["days"] <= 90)).sum())})
    st["fl_share"] = st["fl_n"] / st["n_pay"]
    st = st.reset_index()
    stray = card[~card["key"].isin(set(p["key"])) & card["generic"]]
    cl = pd.DataFrame({"fl_client_n": card[card["generic"]].groupby("client_id").size(), "n_card": card.groupby("client_id").size(),
                       "fl_client_stray_n": stray.groupby("client_id").size(), "fl_client_stray_90": stray[stray["days"] <= 90].groupby("client_id").size()}).fillna(0)
    cl["fl_client_share"] = cl["fl_client_n"] / cl["n_card"]
    c = candidates(streams)
    c["stream_id"] = streams.reset_index(drop=True)["stream_id"].to_numpy()
    c = c.merge(st, on=["client_id", "stream_id"], how="left").merge(cl, left_on="client_id", right_index=True, how="left")
    for col in ["fl_n", "fl_n_90", "fl_share", "fl_client_n", "fl_client_stray_n", "fl_client_stray_90", "fl_client_share"]:
        c[col] = c[col].fillna(0.0)
    grp = c.groupby("client_id")
    c["fl_share_minus_client"] = c["fl_share"] - grp["fl_share"].transform("mean")
    c["fl_share_rank"] = grp["fl_share"].rank(ascending=False, method="min")
    c["fl_share_of_client_fillers"] = (c["fl_n"] / grp["fl_n"].transform("sum")).fillna(0.0)
    c["fl_last_days_minus_client_min"] = c["fl_last_days"] - grp["fl_last_days"].transform("min")
    return race_order(c.drop(columns=["stream_id", "n_pay", "n_card"]), "monthly-slot")


def fit_predict(train_t, train_labels, score_t, score_clients, features):
    present = set(train_t["client_id"])
    keep, target, _ = training_rows(train_t, train_labels, present)
    cols = FEATURE_COLUMNS + features
    m = lgb.LGBMClassifier(**LGBM_PARAMS).fit(train_t.loc[keep, cols], target[keep])
    s = m.predict_proba(score_t[cols])[:, 1]
    proba = soft_race_proba(score_t, s, race_slot(score_t, "monthly-slot").to_numpy(), jitter_scale(score_t), score_clients)
    return proba, s


def main():
    paths = Paths(Path("."))
    results = {}
    with data.training_run():
        tx, labels = _training_data(paths, with_selection=False)
        clients = labels.index
        table = filler_table(tx, clients)
        splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
        for name, feats in VARIANTS.items():
            parts, s_rows = [], []
            for fold, (fit_idx, score_idx) in enumerate(splitter.split(clients, labels.to_numpy())):
                fit_c, score_c = set(map(str, clients[fit_idx])), clients[score_idx]
                tt = table[table["client_id"].isin(fit_c)].reset_index(drop=True)
                ts = table[table["client_id"].isin(set(map(str, score_c)))].reset_index(drop=True)
                proba, s = fit_predict(tt, labels.iloc[fit_idx], ts, score_c, feats)
                proba.insert(0, "fold", fold); parts.append(proba)
                keep, target, _ = training_rows(ts, labels.iloc[score_idx], set(ts["client_id"]))
                s_rows.append(pd.DataFrame({"s": s[keep], "y": target[keep]}))
            oof = pd.concat(parts).reindex(clients).rename_axis("client_id")
            oof.to_csv(OUT / f"oof_filler_{name}.csv")
            sr = pd.concat(s_rows)
            res = {**macro_f1s(oof, labels), "s_auc": round(roc_auc_score(sr["y"], sr["s"]), 4)}
            results[name] = res
            print(name, json.dumps(res), flush=True)
        train_table = table
    # drift of the extra features: train rows vs test rows
    test_tx = data.load_transactions(paths.raw, "test")
    test_table = filler_table(test_tx, pd.Index(sorted(test_tx["client_id"].unique())))
    for name, feats in VARIANTS.items():
        if not feats:
            continue
        x = pd.concat([train_table[feats], test_table[feats]]).reset_index(drop=True)
        y = pd.Series([0] * len(train_table) + [1] * len(test_table))
        p = pd.Series(index=y.index, dtype=float)
        for fit, score in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
            m = lgb.LGBMClassifier(**LGBM_PARAMS).fit(x.iloc[fit], y.iloc[fit])
            p.iloc[score] = m.predict_proba(x.iloc[score])[:, 1]
        results[name]["drift_auc_train_vs_test"] = round(roc_auc_score(y, p), 4)
        print(f"{name} extra features train-vs-test AUC {results[name]['drift_auc_train_vs_test']}", flush=True)
    for f in RELATIVE + STREAM_ABS + CLIENT_ABS:
        print(f"  {f:32s} train mean {train_table[f].mean():8.3f}  test mean {test_table[f].mean():8.3f}")
    (OUT / "filler_survival.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
