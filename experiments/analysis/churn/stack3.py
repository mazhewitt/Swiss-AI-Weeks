"""Stack with the drift-hardened hidden-series marker (scattered-MCC clusters), and a train-on-clean /
score-on-simulated-shift check of the hidden-series features alone. Writes stack3_results.csv."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import importlib.util
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from recurring_family import decision
from recurring_family.evaluation import score
from recurring_family.config import LABELS, MERCHANT_FAMILIES
OUT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("stk", OUT / "stack_lib.py"); stk = importlib.util.module_from_spec(spec); spec.loader.exec_module(stk)
L, y, C, oof, labels = stk.L, stk.y, stk.C, stk.oof, stk.labels
S2 = pickle.load(open(OUT / "stack2_preds.pkl", "rb")); OWN, ROBUST = S2["OWN"], S2["ROBUST"]
SIM = pickle.load(open(OUT / "simulated_markers.pkl", "rb"))
for c in ["sc_max_n", "sc_n2", "hidden_max_n", "hidden_n2"]:
    L["sim0_" + c] = SIM[0.0][c].reindex(L.index).to_numpy()
SC = ["sim0_sc_max_n", "sim0_sc_n2"]
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
stk.L = L
base = roc_auc_score(y, L.oof_none); rows = []
for name, feats in {"G A + robust + scattered-MCC hidden series": ["oof_none"] + OWN + ROBUST + SC,
                    "H A + scattered-MCC hidden series": ["oof_none"] + OWN + SC}.items():
    p = stk.oof_pred("lgbm", feats, None); rep = np.mean([roc_auc_score(y, stk.oof_pred("lgbm", feats, s)) for s in (1, 2)])
    t, n = f1s(with_none(P0, L.index, p))
    rows.append(dict(variant=name, n_features=len(feats), auc=roc_auc_score(y, p), auc_repeats=rep, auc_gain=roc_auc_score(y, p) - base, tuned_f1=t, nested_f1=n))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT / "stack3_results.csv", index=False); print(R.round(4).to_string(index=False))
# shift check: logistic on none-prob + marker, fitted on clean folds, scored on the q=0.33 simulated marker
from sklearn.linear_model import LogisticRegression
for col in ["hidden_max_n", "sc_max_n"]:
    clean = SIM[0.0][col].reindex(L.index).to_numpy(); shifted = SIM[0.33][col].reindex(L.index).to_numpy()
    pc, ps = np.zeros(len(L)), np.zeros(len(L))
    for tr, te in stk.splits(None):
        X = np.c_[L.logit_none, np.minimum(clean, 4)]
        m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        pc[te] = m.predict_proba(X[te])[:, 1]; ps[te] = m.predict_proba(np.c_[L.logit_none, np.minimum(shifted, 4)][te])[:, 1]
    print(col, "clean AUC", round(roc_auc_score(y, pc), 4), "shifted-marker AUC", round(roc_auc_score(y, ps), 4),
          "mean p(none) family clients clean/shifted", round(pc[y == 0].mean(), 3), round(ps[y == 0].mean(), 3))
