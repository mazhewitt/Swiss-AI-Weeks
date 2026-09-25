"""Race rows (ADR 0002) + extra per-stream features from member payments and history (train only)."""
import pandas as pd, numpy as np
CUT=pd.Timestamp("2026-01-01",tz="UTC"); D=pd.Timedelta(days=1)
c=pd.read_pickle("cache/cand_proj.pkl"); paid=pd.read_pickle("cache/paid.pkl"); tx=pd.read_pickle("cache/tx.pkl")
lab=pd.read_pickle("cache/lab.pkl").set_index("client_id")["target_next_recurring_merchant"]
tx["client_id"]=tx.client_id.astype(str); paid["client_id"]=paid.client_id.astype(str)
cp=tx[(tx.type=="card_payment")&(tx.direction=="out")]
m=paid.merge(cp[["client_id","timestamp","amount","currency","mcc","description","fee"]],on=["client_id","timestamp","amount"],how="left").drop_duplicates(["client_id","stream_id","timestamp","amount"])
print("unmatched member payments",m.currency.isna().mean())
m=m.sort_values(["client_id","stream_id","timestamp"])
def agg(g):
    a=g.amount.to_numpy(); t=g.timestamp
    r={}
    r["cur"]=g.currency.mode().iat[0]; r["n_cur"]=g.currency.nunique(); r["n_desc"]=g.description.nunique(); r["n_mcc"]=g.mcc.nunique()
    r["fee_share"]=(g.fee>0).mean(); r["last_fee"]=float(g.fee.iat[-1]>0)
    r["last_amt_ratio"]=a[-1]/np.median(a); r["amt_change_last"]=float(abs(a[-1]/a[-2]-1)>0.02) if len(a)>1 else np.nan
    r["n_amt_levels"]=len(np.unique(np.round(a,2)))
    r["amt_trend"]=(a[-1]-a[0])/np.median(a)
    r["last_desc_new"]=float(g.description.iat[-1] not in set(g.description.iloc[:-1])) if len(g)>1 else np.nan
    r["last_refunded"]=float(g.refunded.iat[-1]); r["n_refunded"]=g.refunded.sum()
    r["dom_last"]=t.iat[-1].day; r["wd_last"]=t.iat[-1].weekday(); r["hour_last"]=t.iat[-1].hour
    r["hour_med"]=np.median(t.dt.hour)
    gaps=np.diff((t-CUT)/D) if len(t)>1 else np.array([np.nan])
    r["last_gap"]=gaps[-1]; r["max_gap"]=np.nanmax(gaps)
    r["cents"]=float(np.median(np.round(a*100)%100))
    r["desc"]=g.description.mode().iat[0]
    return pd.Series(r)
X=m.groupby(["client_id","stream_id"]).apply(agg).reset_index()
c=c.merge(X,on=["client_id","stream_id"],how="left")
c["overdue"]=c.days_since_last/c.period_days
c["last_gap_ratio"]=c.last_gap/c.period_days
# client-level from history
g=tx.groupby("client_id")
cl=pd.DataFrame({"n_tx":g.size(),"salary":tx[tx.description=="salary"].groupby("client_id").amount.median(),
   "last_tx_days":(CUT-g.timestamp.max())/D, "n_refund":tx[tx.type=="refund"].groupby("client_id").size(),
   "n_card":cp.groupby("client_id").size(),"client_cur":g.currency.agg(lambda s:s.mode().iat[0]),
   "fee_rate":g.fee.apply(lambda f:(f>0).mean())})
# non-stream subscription-like card payments (card payments not in any stream, non-retail descriptions)
retail={"fresh foods","pharmacy","hotel booking","electronics shop","ride share","coffee shop","grocery store","neighborhood market","online marketplace","casual dining"}
key=m[["client_id","timestamp","amount"]].assign(ins=1)
ns=cp.merge(key,on=["client_id","timestamp","amount"],how="left"); ns=ns[ns.ins.isna()]
ns=ns[~ns.description.str.split().str[:2].str.join(" ").isin(retail) & ~ns.description.isin(retail)]
ns.to_pickle("cache/nonstream_subs.pkl")
cl["n_ns_sub"]=ns.groupby("client_id").size(); cl["n_ns_sub_90"]=ns[ns.timestamp>=CUT-90*D].groupby("client_id").size()
cl=cl.fillna({"n_refund":0,"n_ns_sub":0,"n_ns_sub_90":0})
c=c.merge(cl,left_on="client_id",right_index=True,how="left")
c["cur_is_client"]=(c.cur==c.client_cur).astype(float)
# race rows with original order (days_to_next, NaN last, more payments first)
c["_o"]=c.days_to_next.fillna(np.inf); c=c.sort_values(["client_id","_o","n_payments"],ascending=[True,True,False])
c["y"]=c.client_id.map(lab)
def targets(g):
    y=g.y.iat[0]; t=np.full(len(g),np.nan)
    if y=="none": t[:]=0
    else:
        hit=np.flatnonzero(g.family.astype(str).to_numpy()==y)
        if len(hit): t[:hit[0]]=0; t[hit[0]]=1
    return pd.Series(t,index=g.index)
c["t"]=c.groupby("client_id",group_keys=False).apply(targets)
c.to_pickle("cache/race.pkl"); print(c.t.value_counts(dropna=False))
