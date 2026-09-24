import pandas as pd, numpy as np
from sklearn.metrics import f1_score
C=pd.Timestamp('2026-01-01',tz='UTC')
def prep(s):
    s=s[s.grp!='other'].copy()
    s['fam']=np.where(s.family=='sm?',np.where(s.amed<15.5,'music','streaming'),s.family)
    return s
def predict(s,clients,minn=3,rule='soonest',H=90,act=1.6,roll=True):
    r=s[s.n>=minn].copy()
    per=r.per.clip(lower=7)
    nxt=r['last']+pd.to_timedelta(per,unit='D')
    if roll:
        k=np.ceil(((C-nxt).dt.total_seconds()/86400/per).clip(lower=0))
        nxt=nxt+pd.to_timedelta(k*per,unit='D')
    r['nd']=(nxt-C).dt.total_seconds()/86400
    r['active']=r.days_since_last<=act*per+3
    if rule!='anyactive_mostrecent':
        r=r[r.active]
    r=r[r.nd<=H]
    if rule=='soonest': r=r.sort_values('nd')
    elif rule=='mostrecent': r=r.sort_values('last',ascending=False)
    elif rule=='longest': r=r.sort_values('n',ascending=False)
    elif rule=='cheapest': r=r.sort_values('amed')
    elif rule=='priciest': r=r.sort_values('amed',ascending=False)
    elif rule=='latest_first': r=r.sort_values('first',ascending=False)
    elif rule=='earliest_first': r=r.sort_values('first')
    p=r.groupby('client_id').fam.first()
    return p.reindex(clients).fillna('none')
def score(y,p):
    return (y.values==p.values).mean(), f1_score(y,p,average='macro')
