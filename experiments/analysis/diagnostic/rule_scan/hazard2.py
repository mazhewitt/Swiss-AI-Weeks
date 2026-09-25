"""On clean amount-cluster streams: what predicts that the soonest alive stream is the label?"""
import pandas as pd, numpy as np
pd.set_option('display.width',250)
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
K=pd.read_pickle(S+'K_0.02.pkl')
k=K[(K.n>=3)&(K.cv<0.03)&K.med_gap.between(10,100)&(K.last_days<=1.3*K.med_gap)].copy()
k['y']=k.client_id.map(y); k['hit']=k.fam==k.y; k['none']=k.y=='none'
s=k.sort_values('next_days').groupby('client_id').head(1)
nn=k[k.y!='none']; nn=nn[nn.client_id.isin(nn[nn.hit].client_id)]
print('oracle-none accuracy of orderings among clients with label in alive clean streams (N=%d):'%nn.client_id.nunique())
for col,asc in [('next_days',1),('first_days',0),('first_days',1),('n',1),('last_days',0),('last_days',1)]:
    print(f'  {col} asc={asc}: {nn.sort_values(col,ascending=bool(asc)).groupby("client_id").head(1).hit.mean():.3f}')
print('soonest alive stream: hit',s.hit.mean().round(3),'none',s.none.mean().round(3))
for col in ['n','first_days','next_days','gap_mad','cv','med_gap','purity']:
    b=pd.qcut(s[col],5,duplicates='drop'); print('--',col); print(s.groupby(b,observed=True)[['hit','none']].mean().round(3).assign(N=s.groupby(b,observed=True).size()).to_string())
