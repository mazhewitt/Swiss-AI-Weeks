"""Which next-date model fits streams better, backtested on history: last+median gap vs anchor day-of-month vs first + k*period."""
import pandas as pd, numpy as np
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
c=pd.read_pickle(S+'tagged.pkl'); c=c[c.fam.notna()&~c.decoy]
base=c.description.str.replace(r'^(billing|pay|member) ','',regex=True)
errs=[]
for (cl,cur,f),g in c.groupby(['client_id','currency','fam']):
    g=g.sort_values('amount'); 
    med=g.amount.median(); h=g[(g.amount/med-1).abs()<0.03].sort_values('timestamp')
    if len(h)<6: continue
    t=h.timestamp.values.astype('datetime64[s]').astype(float)/86400
    gaps=np.diff(t)
    if not (25<np.median(gaps)<35): continue
    tr,te=t[:-1],t[-1]
    mg=np.median(np.diff(tr))
    p1=tr[-1]+mg
    # linear fit t = a + b*k  (fixed period from first)
    k=np.arange(len(tr)); b,a=np.polyfit(k,tr,1); p2=a+b*len(tr)
    # anchor dom: monthly calendar: median dom of history, placed in the month after last
    dts=pd.to_datetime(h.timestamp.values[:-1]); dom=int(np.median(dts.day)); last=dts[-1]
    nm=(last+pd.offsets.MonthBegin(1)).tz_localize(None).normalize()
    p3=(nm+pd.Timedelta(days=min(dom,28)-1)).value/1e9/86400
    errs.append(dict(e_lastgap=p1-te,e_linfit=p2-te,e_dom=p3-te,resid=np.std(tr-(a+b*k)),mg=mg,b=b))
E=pd.DataFrame(errs); print(len(E)); print(E.abs().describe().round(2))
print('period b quantiles', E.b.quantile([.05,.25,.5,.75,.95]).round(2).to_dict())
