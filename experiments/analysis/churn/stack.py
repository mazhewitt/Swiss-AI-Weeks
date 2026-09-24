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

MODELS = {
    "ranker OOF none-prob only": ("logit", ["logit_none"]),
    "logit: none-prob + 12 stream-only churn features": ("logit", ["logit_none"] + STREAM_SMALL),
    "LGBM: none-prob + all stream-only features": ("lgbm", ["oof_none"] + STREAM),
    "logit: stream-only features alone (no none-prob)": ("logit", STREAM_SMALL),
    "logit: none-prob + hidden-series (raw)": ("logit", ["logit_none"] + HIDDEN),
    "logit: none-prob + stream-only + hidden-series": ("logit", ["logit_none"] + STREAM_SMALL + HIDDEN),
    "LGBM: none-prob + stream-only + hidden + other raw": ("lgbm", ["oof_none"] + STREAM + HIDDEN + RAW_OTHER),
}
rows, preds = [], {}
base_auc = roc_auc_score(y, L.oof_none)
for name, (kind, feats) in MODELS.items():
    aucs = []
    for seed in [None, 1, 2, 3, 4]:
        p = oof_pred(kind, feats, seed)
        if seed is None: preds[name] = p
        aucs.append(roc_auc_score(y, p))
    # paired bootstrap of the gain on the ranker-fold predictions
    rng = np.random.default_rng(0); g = []
    for _ in range(500):
        i = rng.integers(0, len(y), len(y)); g.append(roc_auc_score(y[i], preds[name][i]) - roc_auc_score(y[i], L.oof_none.to_numpy()[i]))
    rows.append(dict(model=name, n_features=len(feats), auc_ranker_folds=aucs[0], auc_mean_5_repeats=np.mean(aucs),
                     gain_vs_ranker=aucs[0] - base_auc, gain_ci_lo=np.percentile(g, 2.5), gain_ci_hi=np.percentile(g, 97.5)))
R = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(f"live-stream Clients {len(L)}, none rate {y.mean():.3f}; ranker OOF none AUC {base_auc:.4f}, (1 - max family) AUC {roc_auc_score(y, L.oof_1_minus_max):.4f}")
print(R.round(4).to_string(index=False))
R.to_csv(OUT / "stack_results.csv", index=False)

# ---- Task 5: OOF tuned macro-F1 -------------------------------------------------------
P0 = oof[list(LABELS)].copy(); lab = labels.reindex(P0.index)
def tuned(P):
    d, s = decision.fit(P, lab); return s["tuned_macro_f1"], s["argmax_macro_f1"], d
def with_none(P, idx, pnew):
    P = P.copy(); old = P.loc[idx, "none"].to_numpy()
    fam = P.loc[idx, list(MERCHANT_FAMILIES)].to_numpy()
    scale = np.where(old < 1, (1 - pnew) / np.clip(1 - old, 1e-9, None), 0.0)
    P.loc[idx, list(MERCHANT_FAMILIES)] = fam * scale[:, None]; P.loc[idx, "none"] = pnew
    return P
def nested(P):
    """decision layer fitted on 4 ranker folds, applied to the 5th: an honest macro-F1."""
    from recurring_family.evaluation import score
    pred = pd.Series(index=P.index, dtype=object)
    for k in range(5):
        tr = oof.fold != k
        d, _ = decision.fit(P[tr], lab[tr]); pred[~tr] = d.apply(P[~tr])
    return score(lab, pred)["macro_f1"]
f1 = []
b, a, d0 = tuned(P0); f1.append(dict(variant="baseline ranker OOF", tuned_macro_f1=b, argmax_macro_f1=a, nested_macro_f1=nested(P0)))
print("baseline decision:", d0.weights, d0.none_threshold)
live_none = L.index[L.none == 1]
Pc = with_none(P0, live_none, np.ones(len(live_none)))
b, a, _ = tuned(Pc); f1.append(dict(variant="ceiling: live-stream truth-none set to none=1", tuned_macro_f1=b, argmax_macro_f1=a, nested_macro_f1=nested(Pc)))
for name in ["logit: none-prob + 12 stream-only churn features", "LGBM: none-prob + all stream-only features",
             "logit: none-prob + hidden-series (raw)", "logit: none-prob + stream-only + hidden-series",
             "LGBM: none-prob + stream-only + hidden + other raw"]:
    Ps = with_none(P0, L.index, preds[name])
    b, a, _ = tuned(Ps); f1.append(dict(variant="stack: " + name, tuned_macro_f1=b, argmax_macro_f1=a, nested_macro_f1=nested(Ps)))
F = pd.DataFrame(f1); F["gain_tuned"] = F.tuned_macro_f1 - F.tuned_macro_f1.iloc[0]; F["gain_nested"] = F.nested_macro_f1 - F.nested_macro_f1.iloc[0]
print(F.round(4).to_string(index=False))
F.to_csv(OUT / "f1_results.csv", index=False)
pickle.dump(dict(preds=preds, idx=L.index), open(OUT / "stack_preds.pkl", "wb"))
