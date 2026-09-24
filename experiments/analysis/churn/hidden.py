"""Hidden streams: amount clusters (tol 0.06) of orphan subscription-like payments. Does their presence,
recency or projected next payment (vs the soonest live family stream) explain none?"""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from recurring_family.streams import _evidence
from recurring_family.config import CUTOFF
OUT = Path(__file__).parent
D = pickle.load(open(OUT / "prep.pkl", "rb")); C = pickle.load(open(OUT / "client_feats.pkl", "rb"))
tx, st, members, lab = D["tx"], D["streams"], D["members"], D["labels"]
DAY = pd.Timedelta(days=1)
in_stream = set(np.concatenate([m["idx"] for m in members]))
cp = tx[tx.type == "card_payment"].copy()
cp["kind"] = [_evidence(d, m)[0] for d, m in zip(cp.description, cp.mcc)]
o = cp[(cp.kind != "drop") & ~cp.index.isin(in_stream)].copy()
rows = []
for c, g in o.groupby("client_id"):
    g = g.sort_values("amount"); la = np.log(g.amount.to_numpy())
    br = np.concatenate([[0], np.cumsum(np.diff(la) > 0.06)])
    for k in np.unique(br):
        m = g[br == k].sort_values("timestamp")
        ts = m.timestamp
        gaps = np.diff(ts.to_numpy()).astype("timedelta64[s]").astype(float) / 86400
        rows.append(dict(client_id=c, n=len(m), amount=m.amount.median(), first=ts.min(), last=ts.max(),
                         med_gap=np.median(gaps) if len(gaps) else np.nan, mccs=m.mcc.nunique(),
                         descs=";".join(m.description.astype(str).str.lower().unique()[:4])))
Hs = pd.DataFrame(rows)
Hs["y"] = Hs.client_id.map(lab)
pd.set_option("display.width", 220, "display.max_rows", 100, "display.max_colwidth", 60)
print("hidden clusters with n>=3 by label:", Hs[Hs.n >= 3].groupby(Hs.y.eq("none")).size().to_dict())
print(Hs[Hs.n >= 3].sample(15, random_state=0)[["client_id", "y", "n", "amount", "first", "last", "med_gap", "mccs", "descs"]])
print("median gap of n>=3 hidden clusters:", Hs[Hs.n >= 3].med_gap.describe().round(1).to_dict())
print("first payment month of n>=3 hidden clusters:", Hs[Hs.n >= 3]["first"].dt.to_period("M").value_counts().sort_index().to_dict())
print("n>=2 hidden clusters by label:", Hs[Hs.n >= 2].groupby(Hs.y.eq("none")).size().to_dict())
# per-client summaries
g = Hs.groupby("client_id")
S = pd.DataFrame({"h_max_n": g.n.max(), "h_n2": g.n.apply(lambda s: (s >= 2).sum()), "h_n3": g.n.apply(lambda s: (s >= 3).sum()),
                  "h_last_days": (CUTOFF - g["last"].max()) / DAY}).reindex(C.index)
S = S.fillna({"h_max_n": 0, "h_n2": 0, "h_n3": 0})
S["none"] = C.none; S["live"] = C.n_live > 0; S["oof_none"] = C.oof_none
L = S[S.live]
for f in ["h_max_n", "h_n2", "h_n3"]:
    print(f, "AUC", round(roc_auc_score(L.none, L[f]), 3), L.groupby(L[f].clip(upper=4)).none.agg(["size", "mean"]).round(3).to_dict("index"))
# among live clients WITHOUT any orphan: what does the ranker do, and what is the none rate
no = C[(C.n_live > 0) & (C.raw_orphan_sub_all == 0)]
print("live & no orphan: n", len(no), "none rate", round(no.none.mean(), 3), "ranker AUC", round(roc_auc_score(no.none, no.oof_none), 3))
pickle.dump(S, open(OUT / "hidden_client.pkl", "wb")); pickle.dump(Hs, open(OUT / "hidden_clusters.pkl", "wb"))
print("hidden n>=3 clusters: MCC count dist", Hs[Hs.n >= 3].mccs.value_counts().sort_index().to_dict(),
      "median amount", Hs[Hs.n >= 3].amount.median(), "last payment days before cutoff (median)",
      ((CUTOFF - Hs[Hs.n >= 3]["last"]) / DAY).median())
LN = S[S.live & (S.none == 1)]
print("live truth-none clients", len(LN), "with a hidden cluster n>=2:", int((LN.h_n2 > 0).sum()), "n>=3:", int((LN.h_n3 > 0).sum()))
AN = S[S.none == 1]
print("all truth-none", len(AN), "with n>=2:", int((AN.h_n2 > 0).sum()), " no-live none with n>=2:", int(((AN.h_n2 > 0) & ~AN.live).sum()))
LF = S[S.live & (S.none == 0)]
print("live truth-family with n>=2:", int((LF.h_n2 > 0).sum()), "of", len(LF))
