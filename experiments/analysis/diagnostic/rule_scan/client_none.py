"""Is `none` a client-level event? Check activity trends vs none rate, conditional on having clean active streams."""
import pandas as pd, numpy as np
pd.set_option('display.width',250)
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
tx=pd.read_pickle(S+'tx.pkl'); y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
K=pd.read_pickle(S+'K_0.02.pkl'); CUT=pd.Timestamp('2026-01-01',tz='UTC')
A=K[(K.n>=4)&K.med_gap.between(25,35)&(K.last_days<=1.2*K.med_gap)]
d=pd.DataFrame({'none':(y=='none')})
d['n_act']=A.groupby('client_id').size().reindex(d.index).fillna(0)
d['n_streams_ever']=K[(K.n>=4)&K.med_gap.between(25,35)].groupby('client_id').size().reindex(d.index).fillna(0)
d['n_dead']=d.n_streams_ever-d.n_act
dd=lambda days: tx[tx.timestamp>CUT-pd.Timedelta(days=days)].groupby('client_id').size().reindex(d.index).fillna(0)
d['tx30']=dd(30); d['tx90']=dd(90); d['tx365']=tx.groupby('client_id').size()
d['trend']=d.tx90/(d.tx365/4+1)
sal=tx[tx.description=='salary']; d['last_sal']=(CUT-sal.groupby('client_id').timestamp.max()).dt.days.reindex(d.index)
d['hist_days']=(CUT-tx.groupby('client_id').timestamp.min()).dt.days
d['sub_share']=None
print(pd.crosstab(d.n_act,d.none,margins=True))
print(pd.crosstab([d.n_act.clip(0,3)],d.n_dead.clip(0,3),values=d.none,aggfunc='mean').round(2))
s=d[d.n_act>=1]
for col in ['tx30','trend','last_sal','hist_days','n_dead']:
    print('--',col); k=pd.qcut(s[col],5,duplicates='drop'); print(s.groupby(k,observed=True).none.agg(['mean','size']).round(3))
