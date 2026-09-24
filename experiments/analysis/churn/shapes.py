"""Binned none rate (live-stream train Clients) for the top stack drivers."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).parent
C = pickle.load(open(OUT / "client_feats.pkl", "rb")); L = C[C.n_live > 0]
for f in ["min_amount_cv", "min_gap_recent_minus_all", "max_gap_trend", "stream_first_days", "max_n_payments",
          "max_family_description_share", "max_refund_last_days", "max_refund_rate", "min_amt_last_change", "n_families_live"]:
    x = L[f]
    b = pd.qcut(x.rank(method="first"), 8, labels=False) if x.nunique() > 8 else x
    t = L.groupby(b).agg(lo=(f, "min"), hi=(f, "max"), n=("none", "size"), none=("none", "mean"))
    print(f); print(t.round(3).T.to_string()); print()
