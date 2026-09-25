import pandas as pd, numpy as np
pd.set_option('display.width',250)
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
K=pd.read_pickle(S+'K_0.02.pkl'); F=pd.read_pickle(S+'F3.pkl')
A=K[(K.n>=4)&K.med_gap.between(25,35)&(K.last_days<=1.2*K.med_gap)]
na=y[(~y.index.isin(A.client_id))&(y!='none')]
print('non-none with no clean active stream:',len(na))
Kl=K.merge(na.rename('y'),left_on='client_id',right_index=True); Kl=Kl[Kl.fam==Kl.y]
b=Kl.sort_values('n',ascending=False).groupby('client_id').first()
print('has any amount-cluster (n>=2) of label fam:',len(b))
print(b[['n','cv','med_gap','last_days','first_days']].describe().round(1))
print('n dist',b.n.value_counts().sort_index().to_dict())
Fl=F[F.client_id.isin(na.index)]; Fl=Fl[Fl.fam==Fl.client_id.map(na)]
print('raw fam payments present:',Fl.client_id.nunique(),' first_days of label fam (raw):',Fl.first_days.describe().round(0).to_dict())
