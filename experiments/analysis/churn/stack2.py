"""Decomposition of the stack gain: client-level none modelling on the ranker's OWN fields vs new churn
features (robust / filler-sensitive / hidden-series). AUC on live-stream Clients (ranker folds + 2 repeats),
OOF tuned and nested macro-F1 on all 2000 train Clients. Writes stack2_results.csv."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import importlib.util
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from recurring_family import decision
from recurring_family.evaluation import score
from recurring_family.config import LABELS, MERCHANT_FAMILIES
OUT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("stk", OUT / "stack_lib.py"); stk = importlib.util.module_from_spec(spec); spec.loader.exec_module(stk)
L, y, C, oof, labels = stk.L, stk.y, stk.C, stk.oof, stk.labels
agg = lambda fields: [f"{p}_{f}" for f in fields for p in ("prim", "min", "max")]
OWN = agg(["n_payments", "days_since_first", "days_since_last", "period_days", "gap_mad_days", "amount_cv", "median_amount",
           "refund_rate", "days_to_next"]) + ["n_streams", "n_live"]
ROBUST = agg(["overdue", "last_gap_ratio", "last_gap_missed", "last_gap_late", "n_missed", "missed_rate", "missed_last3",
              "gap_trend", "gap_recent_minus_all", "gap_mad_rel", "expected_minus_actual", "phase", "refund_last_full",
              "refund_last2_full", "refund_last_days", "n_refunds", "last_dom", "first_dom", "next_dom"]) + \
         ["n_ended", "n_short", "past_churn_rate", "n_ended_recent120", "n_families_live", "n_families_all", "total_payments",
          "stream_first_days", "stream_last_days", "stream_activity_ratio60", "stream_activity_ratio90"] + \
         [c for c in C.columns if c.startswith("live_has_")]
FILLER = agg(["family_description_share", "last_desc_specific", "last_desc_diff", "amt_last_ratio", "amt_last_logdev",
              "amt_last_change", "amt_last_up", "amt_trend", "amt_first_last", "amt_n_changes", "amt_last3_changes"])
HIDDEN = ["raw_hidden_max_n", "raw_hidden_n2", "raw_hidden_n3", "raw_hidden_last_days"]
VARIANTS = {
    "A own ranker fields, client-level": ["oof_none"] + OWN,
    "B A + robust churn features": ["oof_none"] + OWN + ROBUST,
    "C B + filler-sensitive features": ["oof_none"] + OWN + ROBUST + FILLER,
    "D B + hidden series (detector change)": ["oof_none"] + OWN + ROBUST + HIDDEN,
    "E hidden series only (+ none-prob)": ["oof_none"] + HIDDEN,
    "F all stream + hidden": ["oof_none"] + OWN + ROBUST + FILLER + HIDDEN,
}
P0 = oof[list(LABELS)].copy(); lab = labels.reindex(P0.index)
def with_none(P, idx, pnew):
    P = P.copy(); old = P.loc[idx, "none"].to_numpy(); fam = P.loc[idx, list(MERCHANT_FAMILIES)].to_numpy()
    scale = np.where(old < 1, (1 - pnew) / np.clip(1 - old, 1e-9, None), 0.0)
    P.loc[idx, list(MERCHANT_FAMILIES)] = fam * scale[:, None]; P.loc[idx, "none"] = pnew; return P
def f1s(P):
    _, s = decision.fit(P, lab)
    pred = pd.Series(index=P.index, dtype=object)
    for k in range(5):
        tr = oof.fold != k; d, _ = decision.fit(P[tr], lab[tr]); pred[~tr] = d.apply(P[~tr])
    return s["tuned_macro_f1"], score(lab, pred)["macro_f1"]
base_auc = roc_auc_score(y, L.oof_none); bt, bn = f1s(P0)
rows = [dict(variant="ranker OOF (baseline)", n_features=1, auc=base_auc, auc_repeats=base_auc, auc_gain=0, tuned_f1=bt, nested_f1=bn)]
preds = {}
for name, feats in VARIANTS.items():
    p = stk.oof_pred("lgbm", feats, None); preds[name] = p
    rep = np.mean([roc_auc_score(y, stk.oof_pred("lgbm", feats, s)) for s in (1, 2)])
    t, n = f1s(with_none(P0, L.index, p))
    rows.append(dict(variant=name, n_features=len(feats), auc=roc_auc_score(y, p), auc_repeats=rep,
                     auc_gain=roc_auc_score(y, p) - base_auc, tuned_f1=t, nested_f1=n))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); R["tuned_gain"] = R.tuned_f1 - bt; R["nested_gain"] = R.nested_f1 - bn
pd.set_option("display.width", 220); print(R.round(4).to_string(index=False)); R.to_csv(OUT / "stack2_results.csv", index=False)
pickle.dump(dict(preds=preds, idx=L.index, OWN=OWN, ROBUST=ROBUST, FILLER=FILLER, HIDDEN=HIDDEN), open(OUT / "stack2_preds.pkl", "wb"))
