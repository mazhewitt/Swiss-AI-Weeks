import pandas as pd, numpy as np
pd.set_option('display.width',250)
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
import sys; sys.path.insert(0,'experiments/analysis/diagnostic/rule_scan')
from features import tag, EVERYDAY, CUT
tx=pd.read_pickle(S+'tx.pkl'); y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
print(pd.crosstab(tx.currency, tx.fee>0, normalize='index').round(3))
print(pd.crosstab(tx.type, tx.fee>0, normalize='index').round(3))
r=tx[tx.type=='refund'].copy(); r['type']='card_payment'  # reuse tagger
rt=tag(r); rt=rt[rt.fam.notna()]
print('tagged sub refunds',len(rt), rt.fam.value_counts().to_dict())
c=pd.read_pickle(S+'tagged.pkl'); c=c[c.fam.notna()]
# does refund follow a payment of same fam within 10d?
m=rt.merge(c[['client_id','fam','timestamp','amount']],on=['client_id','fam'],suffixes=('','_p'))
m['dt']=(m.timestamp-m.timestamp_p).dt.total_seconds()/86400
m=m[(m.dt>0)&(m.dt<15)]
print('refund lag days', m.groupby(level=0).dt.min().describe().round(2).to_dict() if False else m.dt.describe().round(2).to_dict())
# per client-family: last payment refunded?
F=pd.read_pickle(S+'F.pkl')
last=c.sort_values('timestamp').groupby(['client_id','fam']).timestamp.last().rename('lastp').reset_index()
rr=rt.merge(last,on=['client_id','fam']); rr['d']=(rr.timestamp-rr.lastp).dt.total_seconds()/86400
flag=rr[(rr.d>=0)&(rr.d<15)].groupby(['client_id','fam']).size().rename('last_refunded')
anyref=rt.groupby(['client_id','fam']).size().rename('n_ref')
F=F.set_index(['client_id','fam']).join(flag).join(anyref).fillna({'last_refunded':0,'n_ref':0}).reset_index()
F['is_label']=F.apply(lambda r:y.get(r.client_id)==r.fam,axis=1)
act=F[F.last_days<=35]
print('active (last<=35d) streams: P(label==fam) by last_refunded:'); print(act.groupby(act.last_refunded>0).is_label.agg(['mean','size']))
print('by n_ref>0:'); print(act.groupby(act.n_ref>0).is_label.agg(['mean','size']))
F.to_pickle(S+'F2.pkl')
