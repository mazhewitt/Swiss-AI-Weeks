import pandas as pd, numpy as np
from sklearn.metrics import f1_score
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
L=['cloud','gym','insurance','mobile','music','software','streaming','none']
y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
K=pd.read_pickle(S+'K_0.02.pkl')
k=K[(K.n>=3)&(K.cv<0.02)]
print(pd.cut(k.med_gap,[0,5,10,12,16,20,25,35,45,55,65,80,100,200,400]).value_counts().sort_index().to_dict())
print(k.assign(p=pd.cut(k.med_gap,[0,20,45,100,400])).groupby(['p','fam'],observed=True).size().unstack().fillna(0).astype(int))
def sc(p): return f1_score(y,p.reindex(y.index).fillna('none'),labels=L,average='macro')
for k_alive in [1.2,1.5,2.0]:
  for gl in [(10,40),(10,100),(5,200)]:
    d=K[(K.n>=3)&(K.cv<0.02)&K.med_gap.between(*gl)&(K.last_days<=k_alive*K.med_gap)]
    p=d.sort_values('next_days').groupby('client_id').fam.first()
    print(f'gap{gl} alive<={k_alive}: {sc(p):.4f}')
