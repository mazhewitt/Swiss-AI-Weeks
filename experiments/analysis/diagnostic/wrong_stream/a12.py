"""Non-stream payments / refunds naming a family: do they point at the true family in the bucket?"""
import pandas as pd, numpy as np
from recurring_family.streams import _evidence, HOME_MCC
lab=pd.read_pickle("cache/lab.pkl").set_index("client_id")["target_next_recurring_merchant"]
ns=pd.read_pickle("cache/nonstream_subs.pkl"); tx=pd.read_pickle("cache/tx.pkl"); tx["client_id"]=tx.client_id.astype(str)
P=pd.read_csv("cache/P_base.csv",index_col=0); pr=P.idxmax(axis=1)
B=pd.read_csv("cache/bucket_base.csv").iloc[:,0]
def fams(desc,mcc):
    k,f,h=_evidence(desc,mcc)
    f=set(f)
    if "music" in f or "streaming" in f or any("music" in x or "stream" in x for x in f): pass
    if not f and HOME_MCC.get(mcc): f={HOME_MCC[mcc]}
    return f
ns["fams"]=[fams(d,m) for d,m in zip(ns.description,ns.mcc)]
ref=tx[tx.type=="refund"].copy(); ref["fams"]=[fams(d,m) for d,m in zip(ref.description,ref.mcc)]
print("family groups seen:", set().union(*ns.fams))
CUT=pd.Timestamp("2026-01-01",tz="UTC")
for name,src in [("nonstream pay",ns),("refund",ref),("nonstream pay last60d",ns[ns.timestamp>=CUT-pd.Timedelta(days=60)])]:
    per=src.groupby("client_id").fams.apply(lambda s:[f for x in s for f in x])
    t=np.mean([lab[c] in per.get(c,[]) for c in B]); p=np.mean([pr[c] in per.get(c,[]) for c in B])
    both=np.mean([(lab[c] in per.get(c,[])) and (pr[c] not in per.get(c,[])) for c in B])
    rev=np.mean([(lab[c] not in per.get(c,[])) and (pr[c] in per.get(c,[])) for c in B])
    print(f"{name:22s} bucket: names true {t:.3f}  names picked {p:.3f}  true-only {both:.3f}  picked-only {rev:.3f}")
