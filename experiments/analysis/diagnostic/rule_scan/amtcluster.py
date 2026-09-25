"""Amount-cluster streams: per client+currency, group subscription-like card payments whose amounts are within tol of each other."""
import pandas as pd, numpy as np, sys
from sklearn.metrics import f1_score
sys.path.insert(0,'experiments/analysis/diagnostic/rule_scan')
from features import EVERYDAY, DECOY, CUT
pd.set_option('display.width',250)
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
L=['cloud','gym','insurance','mobile','music','software','streaming','none']
c=pd.read_pickle(S+'tagged.pkl'); y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
base=c.description.str.replace(r'^(billing|pay|member) ','',regex=True).str.replace(r' (plus|online|digital|core|service)$','',regex=True)
c=c[~c.decoy & ~base.isin(EVERYDAY|{'foods','shop','booking','market','dining','store','share'})].copy()
def clusters(tol):
    rows=[]
    for (cl,cur),g in c.groupby(['client_id','currency']):
        g=g.sort_values('amount'); a=g.amount.values; cid=np.zeros(len(a),int); k=0
        for i in range(1,len(a)):
            if a[i]/a[i-1]-1>tol: k+=1
            cid[i]=k
        g=g.assign(cid=cid)
        for _,h in g.groupby('cid'):
            if len(h)<2: continue
            h=h.sort_values('timestamp'); t=h.timestamp
            fams=h.fam.dropna(); fam=fams[h.kw.loc[fams.index]].mode() if h.kw.any() else fams.mode()
            if not len(fam): continue
            gaps=t.diff().dt.total_seconds().div(86400).dropna()
            ld=(CUT-t.iloc[-1]).total_seconds()/86400
            rows.append(dict(client_id=cl,fam=fam.iloc[0],n=len(h),cv=h.amount.std()/h.amount.mean(),med_gap=gaps.median(),
                gap_mad=(gaps-gaps.median()).abs().median(),last_days=ld,next_days=gaps.median()-ld,first_days=(CUT-t.iloc[0]).days,
                purity=(h.fam==fam.iloc[0]).mean()))
    return pd.DataFrame(rows)
def sc(p): return f1_score(y,p.reindex(y.index).fillna('none'),labels=L,average='macro')
best=None
for tol in [0.01,0.02,0.03,0.05]:
    K=clusters(tol)
    for minn in [3,4]:
        for gapok in [(20,40),(25,35)]:
            for k in [1.2,1.5,1.8]:
                d=K[(K.n>=minn)&K.med_gap.between(*gapok)&(K.last_days<=k*K.med_gap)]
                p=d.sort_values('next_days').groupby('client_id').fam.first()
                s=sc(p); print(f'tol={tol} n>={minn} gap{gapok} alive<={k}*gap -> {s:.4f}')
                if best is None or s>best[0]: best=(s,tol,minn,gapok,k)
    K.to_pickle(S+f'K_{tol}.pkl')
print('BEST',best)
