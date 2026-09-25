import pandas as pd, numpy as np, re
pd.set_option('display.width',250); pd.set_option('display.max_rows',100)
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
tx=pd.read_pickle(S+'tx.pkl'); F=pd.read_pickle(S+'F3.pkl'); c=pd.read_pickle(S+'tagged.pkl')
y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
print('after cutoff:',(tx.timestamp>=pd.Timestamp('2026-01-01',tz='UTC')).sum())
print(tx[tx.description.str.contains('renew|cancel|next|trial|final|last|end|termin|expir|annual|year|upgrade|downgrade',regex=True)].description.value_counts().head(20))
toks=tx.description.str.split().explode().value_counts(); print('all tokens:',len(toks)); print(toks.head(80).to_dict())
A=F[(F.last_days<=35)&(F.n>=2)]
one=A[A.n_desc==1]
print(one[['client_id','fam','n','first_days','last_days','amt_med','amt_cv','y']].head(10))
cl=one.client_id.iloc[0]; f=one.fam.iloc[0]
print(c[(c.client_id==cl)&(c.fam==f)][['timestamp','amount','mcc','description']])
