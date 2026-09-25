import pandas as pd, numpy as np
from recurring_family.streams import _evidence, HOME_MCC
d=pd.read_pickle("cache/race3.pkl"); d["family"]=d.family.astype(str)
tx=pd.read_pickle("cache/tx.pkl"); tx["client_id"]=tx.client_id.astype(str)
ref=tx[tx.type=="refund"].copy()
def fams(desc,mcc):
    k,f,h=_evidence(desc,mcc); f=set(f)
    if not f and HOME_MCC.get(mcc): f={HOME_MCC[mcc]}
    return {x for y in f for x in (("music","streaming") if y=="music_or_streaming" else (y,))}
ref["fams"]=[fams(a,b) for a,b in zip(ref.description,ref.mcc)]
ref=ref[ref.fams.map(len)>0].explode("fams")
cnt=ref.groupby(["client_id","fams"]).size().rename("n_fam_refunds")
d=d.merge(cnt,left_on=["client_id","family"],right_index=True,how="left").fillna({"n_fam_refunds":0})
r=d[d.t.notna()&(d.active==1)]
print(r.groupby(r.n_fam_refunds.clip(upper=3)).t.agg(["mean","size"]))
print(r.groupby(r.n_refunded.clip(upper=3)).t.agg(["mean","size"]))
print(r.groupby((r.n_fam_refunds>r.n_refunded)).t.agg(["mean","size"]))
d.to_pickle("cache/race4.pkl")
