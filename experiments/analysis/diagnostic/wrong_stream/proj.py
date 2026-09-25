"""Alternative next-payment projections per stream (train only)."""
import pandas as pd, numpy as np
CUT=pd.Timestamp("2026-01-01",tz="UTC"); D=pd.Timedelta(days=1)
def roll(last, step):
    """first last+k*step (k>=1) at/after cutoff, in days after cutoff"""
    x=(last-CUT)/D+step
    if x<0: x+=step*np.ceil(-x/step)
    return x
def projections(paid, st):
    rows=[]
    for (c,s),g in paid.groupby(["client_id","stream_id"],sort=False):
        t=g.timestamp.sort_values().reset_index(drop=True); n=len(t)
        r={"client_id":c,"stream_id":s}
        last=t.iloc[-1]; ld=(last-CUT)/D
        r["last_d"]=ld
        if n<2:
            rows.append(r); continue
        gaps=np.diff(((t-CUT)/D).to_numpy())
        med=np.median(gaps); base=30.44 if med>20 else (14 if med>10 else 7)
        k=np.maximum(1,np.round(gaps/base)); fg=gaps/k
        r["p_med"]=roll(last,np.median(fg)); r["p_last"]=roll(last,fg[-1]); r["p_mean"]=roll(last,fg.mean())
        r["p_3044"]=roll(last,base)
        # regression on cumulative index (anchored schedule)
        idx=np.concatenate([[0],np.cumsum(k)]); y=((t-CUT)/D).to_numpy()
        if n>=3:
            b,a=np.polyfit(idx,y,1); nk=np.ceil((0-a)/b) if b>0 else idx[-1]+1
            nk=max(nk,idx[-1]+1); r["p_reg"]=a+b*nk
        else: r["p_reg"]=r["p_med"]
        # calendar: same DOM (median of member DOMs), next occurrence >= cutoff, only monthly
        if base>20:
            dom=int(np.median(t.dt.day)); 
            cand=[pd.Timestamp(2026,m,1,tz="UTC")+pd.Timedelta(days=min(dom,pd.Timestamp(2026,m,1).days_in_month)-1) for m in (1,2,3)]
            # must also be >= last + 15 days
            cand=[x for x in cand if (x-last)/D>=15]
            r["p_cal"]=(cand[0]-CUT)/D if cand else np.nan
            dl=t.iloc[-1].day
            cand=[pd.Timestamp(2026,m,1,tz="UTC")+pd.Timedelta(days=min(dl,pd.Timestamp(2026,m,1).days_in_month)-1) for m in (1,2,3)]
            cand=[x for x in cand if (x-last)/D>=15]
            r["p_callast"]=(cand[0]-CUT)/D if cand else np.nan
        else:
            r["p_cal"]=r["p_callast"]=r["p_med"]
        rows.append(r)
    return pd.DataFrame(rows)
if __name__=="__main__":
    paid=pd.read_pickle("cache/paid.pkl"); st=pd.read_pickle("cache/streams.pkl")
    P=projections(paid,st); P.to_pickle("cache/proj.pkl"); print(P.describe().round(1).T)
