"""Label-free train-vs-valid/test shift of the stream features (ticket 11). Reads transactions only.

    uv run python experiments/analysis/filler_streams/shift11.py OUT.csv   (from the checkout whose detector is measured)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from recurring_family import data
from recurring_family.config import MERCHANT_FAMILIES
from recurring_family.none_model import client_features, GROUP_A, GROUP_B
from recurring_family.ranker import candidates, FEATURE_COLUMNS
from recurring_family.streams import detect_stream_payments

RAW = Path("data/raw")
NUMERIC = [c for c in FEATURE_COLUMNS if c != "family"]


def candidate_aggregates(streams):
    rows = candidates(streams)
    num = rows[["client_id", *NUMERIC]]
    agg = num.groupby("client_id").agg(["mean", "min", "max"])
    agg.columns = [f"{a}_{c}" for c, a in agg.columns]
    fam = pd.crosstab(rows["client_id"], rows["family"].astype(str)).reindex(columns=list(MERCHANT_FAMILIES), fill_value=0)
    fam.columns = [f"n_{c}" for c in fam.columns]
    return agg.join(fam)


feat, cand, stats = {}, {}, {}
for split in ("train", "valid", "test"):
    tx = data.load_transactions(RAW, split)
    streams, paid = detect_stream_payments(tx)
    feat[split] = client_features(streams, paid)
    cand[split] = candidate_aggregates(streams)
    n = tx["client_id"].nunique()
    f = feat[split]
    stats[split] = {
        "clients with a stream": len(f) / n,
        "streams per Client": len(streams) / n,
        "Active Streams per Client": streams["active"].sum() / n,
        "max_missed_rate": f["max_missed_rate"].mean(),
        "n_short": f["n_short"].mean(),
        "n_ended": f["n_ended"].mean(),
        "stream payments per Client": len(paid) / n,
    }
    print(split, {k: round(v, 3) for k, v in stats[split].items()}, flush=True)


def adv(a, b):
    x = pd.concat([a, b]).astype(float)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    make = lambda: lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=15, verbose=-1, n_jobs=1)
    aucs = []
    for seed in range(3):
        p = cross_val_predict(make(), x, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
    m = make().fit(x, y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=x.columns).sort_values(ascending=False)
    return float(np.mean(aucs)), ", ".join(imp.index[:4])


out = []
for other in ("valid", "test"):
    for name, frames, cols in (
        ("ranker candidate fields", cand, None),
        ("none model group A", feat, GROUP_A),
        ("none model group B", feat, GROUP_B),
        ("none model A+B", feat, GROUP_A + GROUP_B),
    ):
        a, b = frames["train"], frames[other]
        cols = cols or sorted(set(a.columns) | set(b.columns))
        auc, top = adv(a.reindex(columns=cols), b.reindex(columns=cols))
        print(f"train vs {other}, {name}: AUC {auc:.3f} (top: {top})", flush=True)
        out.append({"measure": f"AUC train vs {other}: {name}", "value": auc, "top": top})
for split, s in stats.items():
    for k, v in s.items():
        out.append({"measure": f"{k} ({split})", "value": v, "top": ""})
pd.DataFrame(out).to_csv(sys.argv[1], index=False)
