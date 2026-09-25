"""Do streams end after a fixed number of payments (contract length)? hit/none of soonest alive stream by payment count n (monthly only)."""
import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',100)
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
K=pd.read_pickle(S+'K_0.02.pkl')
k=K[(K.n>=3)&(K.cv<0.03)&K.med_gap.between(25,35)&(K.last_days<=1.3*K.med_gap)].copy()
k['y']=k.client_id.map(y); k['hit']=k.fam==k.y; k['none']=k.y=='none'
k['age_p']=((k.first_days)/k.med_gap).round()
s=k.sort_values('next_days').groupby('client_id').head(1)
print(s.groupby('n')[['hit','none']].mean().round(2).assign(N=s.groupby('n').size()).T.to_string())
print(s.groupby('age_p')[['hit','none']].mean().round(2).assign(N=s.groupby('age_p').size()).T.to_string())
# single-alive-stream clients only
one=k.groupby('client_id').filter(lambda g:len(g)==1)
print('single alive stream clients',len(one)); print(one.groupby(pd.cut(one.n,[2,4,6,8,10,11,12,13,14,40]),observed=True)[['hit','none']].mean().round(2).assign(N=one.groupby(pd.cut(one.n,[2,4,6,8,10,11,12,13,14,40]),observed=True).size()).to_string())
