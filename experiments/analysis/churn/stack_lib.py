"""Tasks 2 and 5: does a churn model on top of the ranker's OOF none-probability lift the none AUC (live-stream
Clients) and the OOF tuned macro-F1? Stack models are fitted out-of-fold (same 5 folds as the ranker OOF,
plus 4 extra random 5-fold repeats for the AUC). Writes stack_results.csv and f1_results.csv."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from recurring_family import decision
from recurring_family.config import LABELS, MERCHANT_FAMILIES
OUT = Path(__file__).parent
D = pickle.load(open(OUT / "prep.pkl", "rb")); C = pickle.load(open(OUT / "client_feats.pkl", "rb"))
H = pickle.load(open(OUT / "hidden_client.pkl", "rb"))
for c in ["h_max_n", "h_n2", "h_n3", "h_last_days"]: C["raw_hidden_" + c[2:]] = H[c]
oof, labels = D["oof"], D["labels"]
C["logit_none"] = np.log(C.oof_none.clip(1e-4, 1 - 1e-4) / (1 - C.oof_none.clip(1e-4, 1 - 1e-4)))
L = C[C.n_live > 0].copy(); y = L.none.to_numpy()
STREAM = [c for c in C.columns if c.startswith(("prim_", "min_", "max_", "live_has_")) and c != "prim_family"
          and not any(k in c for k in ("refund_last_partial", "n_partial_refunds", "raw_sub_refunds_45d"))] + \
         ["n_streams", "n_live", "n_ended", "n_short", "past_churn_rate", "n_ended_recent120", "n_families_live",
          "n_families_all", "total_payments", "stream_first_days", "stream_last_days", "stream_activity_ratio60", "stream_activity_ratio90"]
# a compact, hand-picked stream-only set for the logistic stack (top univariate, low redundancy)
STREAM_SMALL = ["min_amount_cv", "min_amt_last_change", "total_payments", "stream_first_days", "n_families_live",
                "max_family_description_share", "max_last_desc_specific", "n_ended_recent120", "min_overdue",
                "min_refund_rate", "min_amt_last3_changes", "stream_last_days"]
HIDDEN = ["raw_orphan_sub_all", "raw_orphan_sub_60", "raw_hidden_max_n", "raw_hidden_n2", "raw_hidden_n3"]
RAW_OTHER = ["raw_n_refunds", "raw_refunds_last45", "raw_card_activity_ratio60", "raw_safe_activity_ratio60",
             "raw_n_tx", "raw_first_tx_days", "raw_last_tx_days", "min_n_partial_refunds", "prim_raw_sub_refunds_45d"]

def splits(seed):
    if seed is None:  # the ranker's own OOF folds
        f = L.fold.to_numpy(); return [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in range(5)]
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(L, y))

def oof_pred(kind, feats, seed):
    X = L[feats].astype(float); p = np.zeros(len(L))
    for tr, te in splits(seed):
        if kind == "logit":
            Xt = X.fillna(X.iloc[tr].median())
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(Xt.iloc[tr], y[tr])
            p[te] = m.predict_proba(Xt.iloc[te])[:, 1]
        else:
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=30,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=0, verbose=-1, n_jobs=1)
            m.fit(X.iloc[tr], y[tr]); p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p

