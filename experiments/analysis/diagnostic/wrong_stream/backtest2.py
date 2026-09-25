import pandas as pd, numpy as np
from itertools import combinations
paid=pd.read_pickle("cache/paid.pkl"); paid["client_id"]=paid.client_id.astype(str); paid=paid.sort_values(["client_id","stream_id","timestamp"])
D=pd.Timedelta(days=1); rows=[]
for cutstr in ["2025-09-01","2025-10-01","2025-11-01"]:
    cut=pd.Timestamp(cutstr,tz="UTC")
    for (c,s),g in paid.groupby(["client_id","stream_id"]):
        ts=g.timestamp; t=((ts-cut)/D).to_numpy(); pre=t<0
        if pre.sum()<3 or (~pre).sum()==0 or np.median(np.diff(t[pre]))<20: continue
        tp=t[pre]; lastts=ts[pre].iloc[-1]; actual=t[~pre][0]
        if actual>=45: continue
        k=np.concatenate([[0],np.cumsum(np.maximum(1,np.round(np.diff(tp)/30.44)))])
        r=dict(client_id=c,cut=cutstr,actual=actual,n=len(tp))
        def roll(x,step):
            return x+step*np.ceil(-x/step) if x<0 else x
        for P in [28,29,30,30.44,31]:
            r[f"last+{P}"]=roll(tp[-1]+P,P)
        anc=np.mean(tp-k*30.44); r["anchored30.44"]=roll(anc+(k[-1]+1)*30.44,30.44)
        anc3=np.mean(tp[-3:]-k[-3:]*30.44); r["anchored_last3"]=roll(anc3+(k[-1]+1)*30.44,30.44)
        r["blend"]=0.5*r["last+30.44"]+0.5*r["anchored30.44"]
        # calendar month: same day-of-month as last, next month
        nm=lastts+pd.DateOffset(months=1)
        x=(nm-cut)/D
        while x<0: nm=nm+pd.DateOffset(months=1); x=(nm-cut)/D
        r["calendar_month"]=x
        # time-of-day adds fraction: all above already use exact timestamps
        rows.append(r)
R=pd.DataFrame(rows); M=[c for c in R.columns if c not in ("client_id","cut","actual","n")]
print("n",len(R))
for m in M:
    e=R[m]-R.actual; print(f"{m:16s} MAE {e.abs().mean():.2f} med {e.abs().median():.2f} bias {e.mean():+.2f}", end="")
    ok=tot=0
    for _,g in R.groupby(["client_id","cut"]):
        if len(g)<2: continue
        a=g.actual.to_numpy(); p=g[m].to_numpy()
        for i,j in combinations(range(len(g)),2):
            if a[i]==a[j]: continue
            ok+=(p[i]<p[j])==(a[i]<a[j]); tot+=1
    print(f"  pairwise {ok/tot:.3f}")
# residual structure: actual - (last+30.44) vs previous residual
e=R.actual-R["last+30.44"]; print("sd of last+30.44 error",e.std().round(2))
