import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
d=pd.read_pickle("cache/race2.pkl"); paid=pd.read_pickle("cache/paid.pkl"); paid["client_id"]=paid.client_id.astype(str)
paid=paid.sort_values(["client_id","stream_id","timestamp"])
rows=[]
for (cid,sid),g in paid.groupby(["client_id","stream_id"]):
    a=g.amount.to_numpy(); n=len(a)
    if n<4: continue
    med=np.median(a[:-1]); dev=np.abs(a/med-1)
    rows.append(dict(client_id=cid,stream_id=sid,last_dev=dev[-1],cv_prev=np.std(a[:-1])/np.mean(a[:-1]),
        dev_last3=dev[-3:].mean(), dev_first3=dev[:3].mean(), dev_mid=np.median(dev[:-1]),
        trend_dev=np.corrcoef(np.arange(n),dev)[0,1] if dev.std()>0 else 0,
        roll_ratio=np.abs(np.diff(a[-4:])/a[-4:-1]).mean()/max(1e-6,np.abs(np.diff(a[:-3])/a[:-4]).mean()) if n>=6 else np.nan))
A=pd.DataFrame(rows); d=d.merge(A,on=["client_id","stream_id"],how="left"); d.to_pickle("cache/race3.pkl")
r=d[d.t.notna()&(d.active==1)&d.last_dev.notna()]
for k in ["last_dev","cv_prev","dev_last3","dev_first3","dev_mid","trend_dev","roll_ratio","amount_cv","amount"]:
    v=r[[k,"t"]].dropna(); print(k, round(roc_auc_score(v.t,v[k]),3), len(v))
print(r.groupby(pd.qcut(r.dev_first3,4)).t.mean().round(3).to_dict())
print(r.groupby(pd.qcut(r.dev_last3,4)).t.mean().round(3).to_dict())
print(r.groupby(pd.qcut(r.roll_ratio,5)).t.mean().round(3).to_dict())
