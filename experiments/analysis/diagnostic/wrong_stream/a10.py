import pandas as pd, numpy as np
d=pd.read_pickle("cache/race3.pkl"); d["family"]=d.family.astype(str)
lab=pd.read_pickle("cache/lab.pkl").set_index("client_id")["target_next_recurring_merchant"]
d["y"]=d.client_id.map(lab)
new=(d.n_payments==1)&(d.days_since_last<=62)
d["slot"]=np.where(d.n_payments==1, (-d.days_since_last)%30.44, d.days_to_next)  # monthly slot for singles
d["slot"]=np.where((d.n_payments==1)&(d.days_since_last<=30.44), 30.44-d.days_since_last, d.slot)
cl=d[new].client_id.unique(); print("clients with new single-payment stream (<=62d):",len(cl), " share of all 2000:",len(cl)/2000)
g=d[d.client_id.isin(cl)&((d.active==1)|new)].copy()
g["isnew"]=new[g.index]
res=[]
for cid,h in g.groupby("client_id"):
    h=h.sort_values("slot"); y=h.y.iat[0]
    nf=h[h.isnew].family.iat[0] if h.isnew.any() else None
    newfam=set(h[h.isnew].family)
    pos=np.flatnonzero(h.isnew.to_numpy())[0]
    before=set(h.family.iloc[:pos])
    res.append(dict(cid=cid,y=y,label_new=y in newfam,label_before=y in before and y not in newfam,none=y=="none",
                    n_before=len(before), n_active=int(h.active.sum()), dsl=h[h.isnew].days_since_last.iat[0],
                    label_after=(y not in newfam) and (y!="none") and (y in set(h.family.iloc[pos+1:]))))
R=pd.DataFrame(res)
print(R[["label_new","label_before","label_after","none"]].mean().round(3).to_dict())
print(R.groupby(R.n_before.clip(upper=3))[["label_new","label_before","label_after","none"]].agg("mean").round(3).assign(n=R.groupby(R.n_before.clip(upper=3)).size()))
print(R.groupby(pd.cut(R.dsl,[0,15,31,45,62]))[["label_new","label_before","label_after","none"]].mean().round(3).assign(n=R.groupby(pd.cut(R.dsl,[0,15,31,45,62])).size()))
R.to_pickle("cache/newstream.pkl")
