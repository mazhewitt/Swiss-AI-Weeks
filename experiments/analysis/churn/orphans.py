"""What are the subscription-like card payments that join no stream ('orphans')?"""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from recurring_family.streams import _evidence
from recurring_family.config import CUTOFF
OUT = Path(__file__).parent
D = pickle.load(open(OUT / "prep.pkl", "rb")); C = pickle.load(open(OUT / "client_feats.pkl", "rb"))
tx, st, members, lab = D["tx"], D["streams"], D["members"], D["labels"]
in_stream = set(np.concatenate([m["idx"] for m in members]))
cp = tx[tx.type == "card_payment"].copy()
cp["kind"] = [_evidence(d, m)[0] for d, m in zip(cp.description, cp.mcc)]
cp["in_stream"] = cp.index.isin(in_stream)
cp["none"] = cp.client_id.map(lab).eq("none")
o = cp[(cp.kind != "drop") & ~cp.in_stream]
pd.set_option("display.width", 200, "display.max_rows", 200)
print("orphans", len(o), "by kind", o.kind.value_counts().to_dict())
print(o.groupby("none").size(), cp[(cp.kind!="drop")].groupby("none").size())
print(o.groupby(["kind","none"]).size().unstack())
print(o.description.value_counts().head(30))
print(o.mcc.value_counts().head(15))
print(o.amount.describe())
# time distribution of orphans: months before cutoff
o["mb"] = ((CUTOFF - o.timestamp).dt.days // 30)
print(o.groupby(["mb","none"]).size().unstack().head(16))
# example: a none client with a live stream and many orphans
L = C[(C.n_live > 0)]
ex = L[L.none == 1].sort_values("raw_orphan_sub_all", ascending=False).index[:3]
for c in ex:
    print("\n=== ", c, lab[c])
    print(st[st.client_id == c][["stream_id","family","period_days","median_amount","n_payments","first_payment","last_payment","active","description"]].to_string())
    s = cp[(cp.client_id == c) & (cp.kind != "drop")][["timestamp","amount","mcc","description","kind","in_stream"]]
    print(s.to_string())
