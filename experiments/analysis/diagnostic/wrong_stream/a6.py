import pandas as pd, numpy as np
c=pd.read_pickle("cache/race.pkl"); paid=pd.read_pickle("cache/paid.pkl"); paid["client_id"]=paid.client_id.astype(str)
paid=paid.sort_values(["client_id","stream_id","timestamp"])
def steps(g):
    a=g.amount.to_numpy(); rel=a[1:]/a[:-1]-1
    big=np.flatnonzero(np.abs(rel)>0.02)
    r={"n_steps":len(big)}
    if len(big):
        i=big[-1]; r["last_step_from_end"]=len(a)-1-(i+1)  # 0 = last payment is new level
        r["last_step_rel"]=rel[i]; r["max_abs_rel"]=np.abs(rel).max()
    r["rel_list"]=np.round(rel,3).tolist()[-6:]
    return r
S=pd.DataFrame([dict(client_id=k[0],stream_id=k[1],**steps(g)) for k,g in paid.groupby(["client_id","stream_id"])])
d=c.merge(S,on=["client_id","stream_id"]); d.to_pickle("cache/race2.pkl")
r=d[d.t.notna()&(d.active==1)]
print(r.groupby(r.n_steps.clip(upper=4)).t.agg(["mean","size"]))
s=r[r.n_steps>0]
print(s.groupby(s.last_step_from_end.clip(upper=5)).t.agg(["mean","size"]))
print(s.groupby(pd.cut(s.last_step_rel,[-1,-.1,-.02,0,.02,.05,.1,.2,1])).t.agg(["mean","size"]))
print("abs rel quantiles of non-step moves:")
allrel=np.concatenate(r.rel_list.map(np.array).to_list()); print(np.quantile(np.abs(allrel),[.5,.75,.8,.85,.9,.95,.99]).round(4))
print(r.sample(25,random_state=1)[["family","n_payments","t","rel_list"]].to_string())
