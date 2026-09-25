import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
L=['cloud','gym','insurance','mobile','music','software','streaming','none']
F=pd.read_pickle(S+'F.pkl'); lb=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
y=lb
def sc(pred): p=pred.reindex(y.index).fillna('none'); return f1_score(y,p,labels=L,average='macro')
print('gap quantiles',F.med_gap.quantile([.05,.25,.5,.75,.95]).round(1).to_dict())
F['per']=F.med_gap.round(-1)
print(F[F.n>=3].groupby('fam').med_gap.describe().round(1))
res={}
def pick(df,col,asc=True): return df.sort_values(col,ascending=asc).groupby('client_id').fam.first()
res['all none']=sc(pd.Series(dtype=str))
for minn in [1,2,3]:
  for N in [30,45,60,90,1e9]:
    d=F[(F.n>=minn)&(F.last_days<=N)]
    res[f'most recent, n>={minn}, last<={N}']=sc(pick(d,'last_days'))
    d2=d.dropna(subset=['next_days'])
    res[f'soonest last+medgap, n>={minn}, last<={N}']=sc(pick(d2,'next_days'))
# soonest by day of month: next = next occurrence of last_dom after cutoff (in Jan)
F['dom_next']=F.last_dom
for N in [35,45,60]:
  d=F[(F.n>=2)&(F.last_days<=N)]
  res[f'soonest by last DOM, n>=2, last<={N}']=sc(pick(d,'dom_next'))
# next_days relative to period: only streams still "alive": last_days <= k*med_gap
for k in [1.2,1.5,2.0]:
  d=F[(F.n>=3)&(F.last_days<=k*F.med_gap)]
  res[f'soonest last+medgap, alive k={k}']=sc(pick(d,'next_days'))
  res[f'most recent, alive k={k}']=sc(pick(d,'last_days'))
for k,v in sorted(res.items(),key=lambda x:-x[1]): print(f'{v:.4f}  {k}')
# label vs available families
fams=F[F.n>=2].groupby('client_id').fam.agg(set)
df=pd.DataFrame({'y':y}).join(fams); df['fam']=df.fam.apply(lambda s:s if isinstance(s,set) else set())
df['k']=df.fam.apply(len); df['in']=df.apply(lambda r:r.y in r.fam,axis=1)
print(df.groupby('k').agg(n=('y','size'),none=('y',lambda s:(s=='none').mean()),label_in=('in','mean')))
print('non-none label not in any family set:', ((df.y!='none')&~df['in']).sum())
