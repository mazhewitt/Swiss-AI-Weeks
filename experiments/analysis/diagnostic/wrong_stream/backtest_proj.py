"""Backtest next-payment projection methods at shifted cutoffs using member payments (train only)."""
import pandas as pd, numpy as np
from itertools import combinations
paid=pd.read_pickle("cache/paid.pkl"); paid["client_id"]=paid.client_id.astype(str)
paid=paid.sort_values(["client_id","stream_id","timestamp"])
D=pd.Timedelta(days=1)
def proj(t, cut):
    """t: days (relative to cut) of payments before cut (sorted). returns dict of projected first payment >= 0"""
    gaps=np.diff(t); med=np.median(gaps); base=30.44 if med>20 else (14 if med>10 else 7)
    k=np.maximum(1,np.round(gaps/base)); fg=gaps/k
    def roll(step):
        x=t[-1]+step
        if x<0: x+=step*np.ceil(-x/step)
        return x
    out={"med":roll(np.median(fg)),"mean":roll(fg.mean()),"last":roll(fg[-1]),"base":roll(base)}
    idx=np.concatenate([[0],np.cumsum(k)]); b,a=np.polyfit(idx,t,1)
    nk=max(np.ceil(-a/b),idx[-1]+1); out["reg"]=a+b*nk
    # regression with only last 4 payments
    if len(t)>=4:
        b4,a4=np.polyfit(idx[-4:],t[-4:],1); nk=max(np.ceil(-a4/b4),idx[-1]+1); out["reg4"]=a4+b4*nk
    else: out["reg4"]=out["reg"]
    out["avg_med_reg"]=(out["med"]+out["reg"])/2
    return out
rows=[]
for cutstr in ["2025-09-01","2025-10-01","2025-11-01"]:
    cut=pd.Timestamp(cutstr,tz="UTC")
    for (c,s),g in paid.groupby(["client_id","stream_id"]):
        t=((g.timestamp-cut)/D).to_numpy()
        pre=t[t<0]; post=t[t>=0]
        if len(pre)<3 or len(post)==0: continue
        if np.median(np.diff(pre))<20: continue
        p=proj(pre,cut); p.update(client_id=c,stream_id=s,cut=cutstr,actual=post[0],n=len(pre)); rows.append(p)
R=pd.DataFrame(rows)
# only streams whose actual next is 'on time' (not a missed payment): actual < 45
R=R[R.actual<45]
M=["med","mean","last","base","reg","reg4","avg_med_reg"]
print("n streams",len(R))
for m in M:
    e=R[m]-R.actual; print(f"{m:12s} MAE {e.abs().mean():.2f}  median|e| {e.abs().median():.2f}  bias {e.mean():+.2f}")
# pairwise order accuracy within client & cut
acc={m:[0,0] for m in M}
for (c,cut),g in R.groupby(["client_id","cut"]):
    if len(g)<2: continue
    for i,j in combinations(range(len(g)),2):
        a=g.iloc[i]; b=g.iloc[j]
        if a.actual==b.actual: continue
        tr=a.actual<b.actual
        for m in M: acc[m][0]+= (a[m]<b[m])==tr; acc[m][1]+=1
for m in M: print(f"{m:12s} pairwise order acc {acc[m][0]/acc[m][1]:.3f}  (pairs {acc[m][1]})")
R.to_pickle("cache/backtest.pkl")
