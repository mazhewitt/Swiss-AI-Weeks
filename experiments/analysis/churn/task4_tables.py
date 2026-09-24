"""Task 4 tables: none rate by number of live streams, soonest-live family, stream tenure; marker shares."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).parent
C = pickle.load(open(OUT / "client_feats.pkl", "rb")); SIM = pickle.load(open(OUT / "simulated_markers.pkl", "rb"))
print(C.groupby(C.n_live.clip(upper=4)).none.agg(["size", "mean"]).round(3).T.to_string())
L = C[C.n_live > 0]
print(L.groupby("prim_family").none.agg(["size", "mean"]).round(3).T.to_string())
print(L.groupby(pd.cut(L.stream_first_days, [0, 120, 180, 240, 300, 360, 420])).none.agg(["size", "mean"]).round(3).T.to_string())
print(L.groupby(pd.cut(L.max_n_payments, [0, 3, 4, 5, 7, 10, 40])).none.agg(["size", "mean"]).round(3).T.to_string())
M = SIM[0.0]
for q, M in SIM.items():
    print("q", q, "share all clients sc_max_n>=3:", round((M.sc_max_n >= 3).mean(), 4), "sc_n2>=1:", round((M.sc_n2 >= 1).mean(), 4),
          "hidden_max_n>=3:", round((M.hidden_max_n >= 3).mean(), 4))
M = SIM[0.0]; print("none rate | sc_max_n>=3 (all clients):", round(M.none[M.sc_max_n >= 3].mean(), 3), "n", int((M.sc_max_n >= 3).sum()))
