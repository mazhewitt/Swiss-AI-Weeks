import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
c=pd.read_pickle("cache/race.pkl"); r=c[c.t.notna()]
num=[k for k in r.columns if r[k].dtype.kind in "fi" and k not in ("t","stream_id","_o")]
res=[]
for sub,name in [(r,"all"),(r[r.active==1],"active")]:
    for k in num:
        v=sub[[k,"t"]].dropna()
        if v[k].nunique()<2 or len(v)<200: continue
        a=roc_auc_score(v.t,v[k]); res.append((name,k,round(max(a,1-a),3),round(a,3),len(v)))
R=pd.DataFrame(res,columns=["pool","feat","auc_abs","auc","n"]).sort_values(["pool","auc_abs"],ascending=[True,False])
print(R.groupby("pool").head(25).to_string())
ra=r[r.active==1]
for k in ["cur","desc","family","cur_is_client","client_cur"]:
    t=ra.groupby(k).t.agg(["mean","size"]); print(t[t["size"]>=30].sort_values("mean").round(3).T.to_string())
