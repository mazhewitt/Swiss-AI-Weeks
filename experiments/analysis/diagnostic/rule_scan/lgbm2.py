"""LGBM on amount-cluster stream features (+ raw per-family features)."""
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
L=['cloud','gym','insurance','mobile','music','software','streaming','none']
y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
K=pd.read_pickle(S+'K_0.02.pkl'); F=pd.read_pickle(S+'F3.pkl'); tx=pd.read_pickle(S+'tx.pkl')
CUT=pd.Timestamp('2026-01-01',tz='UTC')
k=K[(K.n>=3)&(K.cv<0.03)&K.med_gap.between(10,100)].copy()
k['alive']=k.last_days<=1.3*k.med_gap
k['rank']=k.assign(nd=k.next_days.where(k.alive)).groupby('client_id').nd.rank(method='first')
k['rel_next']=k.next_days/k.med_gap; k['ratio_last']=k.last_days/k.med_gap
best=k.sort_values(['alive','next_days'],ascending=[False,True]).groupby(['client_id','fam']).first().reset_index()
sc=['n','cv','med_gap','gap_mad','last_days','next_days','first_days','alive','rank','rel_next','ratio_last']
W=best.pivot(index='client_id',columns='fam',values=sc); W.columns=[f's_{f}_{c}' for c,f in W.columns]
W2=F.pivot(index='client_id',columns='fam',values=['n','last_days','amt_cv','n_desc']); W2.columns=[f'r_{f}_{c}' for c,f in W2.columns]
C=pd.DataFrame(index=y.index)
C['n_alive']=k[k.alive].groupby('client_id').size(); C['n_dead']=k[~k.alive].groupby('client_id').size()
C['min_next']=k[k.alive].groupby('client_id').next_days.min()
C['min_relnext']=k[k.alive].groupby('client_id').rel_next.min()
C['tx90']=tx[tx.timestamp>CUT-pd.Timedelta(days=90)].groupby('client_id').size(); C['tx_all']=tx.groupby('client_id').size()
C['trend']=C.tx90/(C.tx_all/4+1)
X=C.join(W).join(W2).reindex(y.index).astype(float)
yy=y.map({l:i for i,l in enumerate(L)}).values
res=[]
for seed in [0,1]:
    oof=np.zeros((len(X),8))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,yy):
        m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,n_jobs=1,verbose=-1,random_state=seed)
        m.fit(X.iloc[tr],yy[tr]); oof[te]=m.predict_proba(X.iloc[te])
    res.append(f1_score(yy,oof.argmax(1),average='macro'))
print('LGBM cluster feats 5fold macroF1 per seed',np.round(res,4))
print(pd.Series(f1_score(yy,oof.argmax(1),average=None,labels=range(8)),index=L).round(3).to_dict())
Xs=X[[c for c in X.columns if not c.startswith('r_')]]
oof=np.zeros((len(X),8))
for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,yy):
    m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,n_jobs=1,verbose=-1).fit(Xs.iloc[tr],yy[tr]); oof[te]=m.predict_proba(Xs.iloc[te])
print('without raw fam feats',round(f1_score(yy,oof.argmax(1),average='macro'),4))
