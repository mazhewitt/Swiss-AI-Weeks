"""Task 5, simple version: 2-feature logistic stacks (ranker none-logit + one churn score), OOF on the ranker folds.
Writes f1_simple.csv."""
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
L, y, oof, labels = stk.L, stk.y, stk.oof, stk.labels
SIM = pickle.load(open(OUT / "simulated_markers.pkl", "rb"))
L["sc_max_n"] = SIM[0.0]["sc_max_n"].reindex(L.index).to_numpy()
L["hidden_max_n_capped"] = np.minimum(L.raw_hidden_max_n, 4); L["sc_max_n_capped"] = np.minimum(L.sc_max_n, 4)
stk.L = L
P0 = oof[list(LABELS)].copy(); lab = labels.reindex(P0.index)
def with_none(P, idx, pnew):
    P = P.copy(); old = P.loc[idx, "none"].to_numpy(); fam = P.loc[idx, list(MERCHANT_FAMILIES)].to_numpy()
    scale = np.where(old < 1, (1 - pnew) / np.clip(1 - old, 1e-9, None), 0.0)
    P.loc[idx, list(MERCHANT_FAMILIES)] = fam * scale[:, None]; P.loc[idx, "none"] = pnew; return P
def f1s(P):
    _, s = decision.fit(P, lab); pred = pd.Series(index=P.index, dtype=object)
    for k in range(5):
        tr = oof.fold != k; d, _ = decision.fit(P[tr], lab[tr]); pred[~tr] = d.apply(P[~tr])
    return s["tuned_macro_f1"], score(lab, pred)["macro_f1"]
bt, bn = f1s(P0); base = roc_auc_score(y, L.oof_none)
rows = [dict(stack="ranker OOF (baseline)", auc=base, tuned_f1=bt, nested_f1=bn)]
for extra in ["min_amount_cv", "total_payments", "hidden_max_n_capped", "sc_max_n_capped"]:
    p = stk.oof_pred("logit", ["logit_none", extra], None); t, n = f1s(with_none(P0, L.index, p))
    rows.append(dict(stack=f"logit(none-logit + {extra})", auc=roc_auc_score(y, p), tuned_f1=t, nested_f1=n)); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R["auc_gain"] = R.auc - base; R["tuned_gain"] = R.tuned_f1 - bt; R["nested_gain"] = R.nested_f1 - bn
print(R.round(4).to_string(index=False)); R.to_csv(OUT / "f1_simple.csv", index=False)
