"""Marginal AUC of each churn-candidate group on top of variant A (client-level model on the ranker's own fields).
Writes marginal.csv."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import importlib.util
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
OUT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("stk", OUT / "stack_lib.py"); stk = importlib.util.module_from_spec(spec); spec.loader.exec_module(stk)
L, y, C = stk.L, stk.y, stk.C
S2 = pickle.load(open(OUT / "stack2_preds.pkl", "rb")); OWN = S2["OWN"]
agg = lambda fields: [f"{p}_{f}" for f in fields for p in ("prim", "min", "max")]
GROUPS = {
    "overdue ratio (days_since_last/period)": agg(["overdue", "phase"]),
    "last payment missed / late": agg(["last_gap_ratio", "last_gap_missed", "last_gap_late", "missed_last3"]),
    "missed payments over life": agg(["n_missed", "missed_rate", "expected_minus_actual"]),
    "gap trend": agg(["gap_trend", "gap_recent_minus_all", "gap_mad_rel"]),
    "amount trend / last-payment amount change": agg(["amt_last_ratio", "amt_last_logdev", "amt_last_change", "amt_last_up", "amt_trend", "amt_first_last", "amt_n_changes", "amt_last3_changes"]),
    "refund near the end (full, matched)": agg(["refund_last_full", "refund_last2_full", "refund_last_days", "n_refunds"]),
    "partial refunds (raw refund rows)": agg(["refund_last_partial", "n_partial_refunds"]),
    "streams ended before Cutoff / past churn": ["n_ended", "n_short", "past_churn_rate", "n_ended_recent120"],
    "client stream age / volume": ["total_payments", "stream_first_days", "stream_last_days", "n_families_live", "n_families_all"],
    "stream activity drop (member payments)": ["stream_activity_ratio60", "stream_activity_ratio90"],
    "family of live streams": [c for c in C.columns if c.startswith("live_has_")],
    "day of month": agg(["last_dom", "first_dom", "next_dom"]),
    "family-description share (filler-sensitive)": agg(["family_description_share", "last_desc_specific", "last_desc_diff"]),
    "raw activity drop (decoy-safe types)": ["raw_safe_activity_ratio60", "raw_safe_activity_ratio30", "raw_last_salary_days"],
    "raw card activity / refunds (decoy-exposed)": ["raw_card_activity_ratio60", "raw_n_refunds", "raw_refunds_last45", "raw_n_tx"],
    "hidden series (orphan filler clusters)": ["raw_hidden_max_n", "raw_hidden_n2", "raw_hidden_n3", "raw_hidden_last_days"],
}
aA = np.mean([roc_auc_score(y, stk.oof_pred("lgbm", ["oof_none"] + OWN, s)) for s in (None, 1, 2)])
rows = []
for g, feats in GROUPS.items():
    a = np.mean([roc_auc_score(y, stk.oof_pred("lgbm", ["oof_none"] + OWN + feats, s)) for s in (None, 1, 2)])
    a0 = np.mean([roc_auc_score(y, stk.oof_pred("lgbm", ["oof_none"] + feats, s)) for s in (None, 1, 2)])
    rows.append(dict(group=g, n_features=len(feats), auc_none_prob_plus_group=a0, gain_over_ranker=a0 - roc_auc_score(y, L.oof_none),
                     auc_A_plus_group=a, gain_over_A=a - aA))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows).sort_values("gain_over_A", ascending=False)
print("A (own fields) AUC", round(aA, 4)); print(R.round(4).to_string(index=False)); R.to_csv(OUT / "marginal.csv", index=False)
