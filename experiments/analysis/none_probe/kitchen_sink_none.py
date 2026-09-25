"""A deliberately rule-breaking `none` probe (train labels only, 5-fold out-of-fold): raw-transaction
features that ADR 0001 excludes, against the stream-table `none` features. Which raw signal, if any, predicts
`none` better than 0.88 AUC, and does it transfer (train-vs-test classifier AUC on the same features)?"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from recurring_family import data
from recurring_family.cli import Paths, _training_data
from recurring_family.config import CUTOFF
from recurring_family.none_model import FEATURE_COLUMNS as STREAM_NONE_FEATURES, client_features
from recurring_family.ranker import _detected
from recurring_family.streams import _decoy_described, _names_family, StreamParams, detect_stream_payments

OUT = Path("experiments/analysis/none_probe")
DAY = pd.Timedelta(days=1)
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, random_state=0, deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1)


def words(desc):
    return re.findall(r"[a-z]+", str(desc).lower())


def raw_features(tx: pd.DataFrame, cutoff=CUTOFF, vocab=None, mccs=None):
    tx = tx.copy()
    tx["days"] = (cutoff - tx["timestamp"]) / DAY
    tx["card"] = tx["type"] == "card_payment"
    tx["decoy"] = tx["description"].map(_decoy_described) & tx["card"]
    tx["names_family"] = tx["description"].map(_names_family)
    tx["generic"] = tx["card"] & ~tx["names_family"] & ~tx["decoy"]
    tx["refund"] = tx["type"].str.contains("refund", case=False) | ((tx["direction"] == "in") & tx["card"])
    g = tx.groupby("client_id")
    f = pd.DataFrame(index=g.size().index)
    f["n_tx"] = g.size(); f["n_card"] = g["card"].sum(); f["n_decoy"] = g["decoy"].sum(); f["n_generic"] = g["generic"].sum()
    f["n_family_named"] = g["names_family"].sum(); f["n_refund"] = g["refund"].sum()
    f["n_desc"] = g["description"].nunique(); f["n_mcc"] = g["mcc"].nunique(); f["n_types"] = g["type"].nunique()
    f["history_days"] = g["days"].max(); f["last_any_days"] = g["days"].min()
    for w in (7, 14, 30, 60, 90):
        recent = tx[tx["days"] <= w].groupby("client_id")
        f[f"n_tx_{w}"] = recent.size().reindex(f.index).fillna(0)
        f[f"n_card_{w}"] = recent["card"].sum().reindex(f.index).fillna(0)
        f[f"n_decoy_{w}"] = recent["decoy"].sum().reindex(f.index).fillna(0)
        f[f"n_generic_{w}"] = recent["generic"].sum().reindex(f.index).fillna(0)
        f[f"n_refund_{w}"] = recent["refund"].sum().reindex(f.index).fillna(0)
        f[f"n_family_named_{w}"] = recent["names_family"].sum().reindex(f.index).fillna(0)
        f[f"amount_out_{w}"] = tx[(tx["days"] <= w) & (tx["direction"] == "out")].groupby("client_id")["amount"].sum().reindex(f.index).fillna(0)
    card = tx[tx["card"]]
    cg = card.groupby("client_id")
    f["last_card_days"] = cg["days"].min().reindex(f.index)
    f["last_family_named_days"] = tx[tx["names_family"]].groupby("client_id")["days"].min().reindex(f.index)
    f["last_decoy_days"] = tx[tx["decoy"]].groupby("client_id")["days"].min().reindex(f.index)
    f["last_generic_days"] = tx[tx["generic"]].groupby("client_id")["days"].min().reindex(f.index)
    f["last_refund_days"] = tx[tx["refund"]].groupby("client_id")["days"].min().reindex(f.index)
    f["card_amount_mean"] = cg["amount"].mean().reindex(f.index); f["card_amount_max"] = cg["amount"].max().reindex(f.index)
    f["fee_sum"] = g["fee"].sum(); f["n_in"] = g.apply(lambda d: (d["direction"] == "in").sum())
    # bag of description words over all card payments and over the last 90 days; last card payment's words
    card_words = card.assign(w=card["description"].map(words)).explode("w").dropna(subset=["w"]).reset_index(drop=True)
    if vocab is None:
        vocab = card_words["w"].value_counts().head(80).index.tolist()
    cw = card_words[card_words["w"].isin(vocab)]
    bag = pd.crosstab(cw["client_id"], cw["w"]).reindex(index=f.index, columns=vocab).fillna(0)
    bag.columns = [f"w_{c}" for c in bag.columns]
    recent_cw = cw[cw["days"] <= 90]
    bag90 = pd.crosstab(recent_cw["client_id"], recent_cw["w"]).reindex(index=f.index, columns=vocab).fillna(0)
    bag90.columns = [f"w90_{c}" for c in bag90.columns]
    last = card.sort_values("days").groupby("client_id").first()
    last_words = last["description"].map(words)
    lastw = pd.DataFrame({f"last_{v}": last_words.map(lambda ws: float(v in ws)) for v in vocab}, index=last.index).reindex(f.index).fillna(0)
    if mccs is None:
        mccs = card["mcc"].value_counts().head(25).index.tolist()
    cm = card[card["mcc"].isin(mccs)]
    mcc_bag = pd.crosstab(cm["client_id"], cm["mcc"]).reindex(index=f.index, columns=mccs).fillna(0)
    mcc_bag.columns = [f"mcc_{c}" for c in mcc_bag.columns]
    rm = cm[cm["days"] <= 90]
    mcc90 = pd.crosstab(rm["client_id"], rm["mcc"]).reindex(index=f.index, columns=mccs).fillna(0)
    mcc90.columns = [f"mcc90_{c}" for c in mcc90.columns]
    last_mcc = pd.DataFrame({f"lastmcc_{m}": (last["mcc"] == m).astype(float) for m in mccs}, index=last.index).reindex(f.index).fillna(0)
    out = pd.concat([f, bag, bag90, lastw, mcc_bag, mcc90, last_mcc], axis=1).astype(float)
    return out, vocab, mccs


def stream_none_features(tx, clients):
    streams, paid = _detected(tx, pd.Index(clients))
    return client_features(streams, paid).reindex(clients)


def stray_features(tx, clients):
    """Ticket 11's signal as it was before strays joined streams: payments on no stream, by kind."""
    streams, paid = detect_stream_payments(tx, CUTOFF, StreamParams(join_strays=False))
    card = tx[tx["type"] == "card_payment"].copy()
    key = card["client_id"].astype(str) + "|" + card["timestamp"].astype(str) + "|" + card["amount"].round(2).astype(str)
    pkey = paid["client_id"].astype(str) + "|" + paid["timestamp"].astype(str) + "|" + paid["amount"].round(2).astype(str)
    card["member"] = key.isin(set(pkey))
    card["days"] = (CUTOFF - card["timestamp"]) / DAY
    card["names_family"] = card["description"].map(_names_family)
    card["decoy"] = card["description"].map(_decoy_described)
    stray = card[~card["member"] & ~card["names_family"] & ~card["decoy"]]
    g = stray.groupby("client_id")
    f = pd.DataFrame(index=pd.Index(clients))
    f["stray_n"] = g.size().reindex(f.index).fillna(0)
    f["stray_n_90"] = stray[stray["days"] <= 90].groupby("client_id").size().reindex(f.index).fillna(0)
    f["stray_last_days"] = g["days"].min().reindex(f.index)
    return f.astype(float)


def oof_auc(x, y, name):
    x = x.replace([np.inf, -np.inf], np.nan)
    p = pd.Series(index=y.index, dtype=float)
    imp = pd.Series(0.0, index=x.columns)
    for fit, score in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
        m = lgb.LGBMClassifier(**PARAMS).fit(x.iloc[fit], y.iloc[fit])
        p.iloc[score] = m.predict_proba(x.iloc[score])[:, 1]
        imp += pd.Series(m.booster_.feature_importance("gain"), index=x.columns)
    a = roc_auc_score(y, p)
    print(f"{name:55s} none AUC {a:.3f}   ({x.shape[1]} features)", flush=True)
    return a, p, imp.sort_values(ascending=False)


def drift_auc(x_train, x_test, name):
    x = pd.concat([x_train, x_test]).replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    y = pd.Series([0] * len(x_train) + [1] * len(x_test))
    p = pd.Series(index=y.index, dtype=float)
    for fit, score in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
        m = lgb.LGBMClassifier(**PARAMS).fit(x.iloc[fit], y.iloc[fit])
        p.iloc[score] = m.predict_proba(x.iloc[score])[:, 1]
    a = roc_auc_score(y, p)
    print(f"{name:55s} train-vs-test AUC {a:.3f}", flush=True)
    return a


def main():
    paths = Paths(Path("."))
    with data.training_run():
        tx, labels = _training_data(paths, with_selection=False)
        clients = labels.index
        y = (labels == "none").astype(int)
        raw, vocab, mccs = raw_features(tx)
        raw = raw.reindex(clients)
        stream = stream_none_features(tx, clients)
        stray = stray_features(tx, clients)
        results = {}
        for name, x in [("streams only (the none model's features)", stream), ("raw kitchen sink", raw), ("stray payments (pre-ticket-11 signal)", stray),
                        ("streams + stray", pd.concat([stream, stray], axis=1)), ("streams + raw", pd.concat([stream, raw], axis=1)),
                        ("streams + raw + stray", pd.concat([stream, raw, stray], axis=1))]:
            a, p, imp = oof_auc(x, y, name)
            results[name] = {"auc": round(a, 4), "top_features": imp.head(15).round(0).to_dict()}
        print("\ntop raw kitchen-sink features by gain:")
        print(pd.Series(results["raw kitchen sink"]["top_features"]).to_string())
        print("\ntop streams+raw+stray features by gain:")
        print(pd.Series(results["streams + raw + stray"]["top_features"]).to_string())
        # single-feature AUCs of the top raw features
        print("\nsingle-feature AUC (raw kitchen sink top 15):")
        for c in list(results["raw kitchen sink"]["top_features"])[:15]:
            v = raw[c].fillna(raw[c].median())
            a = roc_auc_score(y, v); print(f"  {c:30s} {max(a, 1-a):.3f} ({'high' if a >= .5 else 'low'} -> none)   mean none {v[y==1].mean():.2f} vs family {v[y==0].mean():.2f}")
    # drift: the same features on test Clients (transactions only)
    test_tx = data.load_transactions(paths.raw, "test")
    test_clients = pd.Index(sorted(test_tx["client_id"].unique()))
    raw_test, _, _ = raw_features(test_tx, vocab=vocab, mccs=mccs)
    raw_test = raw_test.reindex(test_clients)
    results["drift"] = {
        "raw": round(drift_auc(raw, raw_test, "raw kitchen sink"), 4),
        "streams": round(drift_auc(stream, stream_none_features(test_tx, test_clients), "streams only"), 4),
        "stray": round(drift_auc(stray, stray_features(test_tx, test_clients), "stray payments"), 4),
    }
    (OUT / "kitchen_sink_none.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
