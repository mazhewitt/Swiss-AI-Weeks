"""Univariate AUC for 'truth is none' among train Clients with >= 1 Active Stream. Writes univariate.csv."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
OUT = Path(__file__).parent
C = pickle.load(open(OUT / "client_feats.pkl", "rb"))
H = pickle.load(open(OUT / "hidden_client.pkl", "rb"))
for c in ["h_max_n", "h_n2", "h_n3", "h_last_days"]: C["raw_hidden_" + c[2:]] = H[c]
L = C[C.n_live > 0].copy()
y = L.none.to_numpy()
rng = np.random.default_rng(0)
res = []
for c in L.columns:
    if c in ("y", "none", "fold", "prim_family", "raw_main_ccy"): continue
    x = L[c].astype(float)
    ok = x.notna().to_numpy()
    if ok.sum() < 50 or x[ok].nunique() < 2: continue
    auc = roc_auc_score(y[ok], x[ok])
    # AUC with NaN imputed to median (full group)
    xf = x.fillna(x.median())
    aucf = roc_auc_score(y, xf)
    # bootstrap SE
    bs = []
    for _ in range(200):
        i = rng.integers(0, len(y), len(y)); 
        if y[i].min() == y[i].max(): continue
        bs.append(roc_auc_score(y[i], xf.to_numpy()[i]))
    res.append(dict(feature=c, auc=aucf, auc_nonmissing=auc, strength=abs(aucf - 0.5), se=np.std(bs),
                    n_nonmissing=int(ok.sum()), mean_none=float(x[y == 1].mean()), mean_family=float(x[y == 0].mean())))
R = pd.DataFrame(res).sort_values("strength", ascending=False)
R.to_csv(OUT / "univariate.csv", index=False)
pd.set_option("display.width", 200, "display.max_rows", 300)
print("live clients", len(L), "none rate", y.mean())
print(R.round(3).to_string(index=False))
# categorical: family of primary live stream, main currency
for c in ["prim_family", "raw_main_ccy"]:
    t = L.groupby(c).none.agg(["size", "mean"]); print(t.round(3))
