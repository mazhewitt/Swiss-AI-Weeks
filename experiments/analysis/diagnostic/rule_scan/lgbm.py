import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
L=['cloud','gym','insurance','mobile','music','software','streaming','none']
F=pd.read_pickle(S+'F2.pkl'); tx=pd.read_pickle(S+'tx.pkl')
y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
CUT=pd.Timestamp('2026-01-01',tz='UTC')
F['rank_next']=F.assign(nd=F.next_days.where(F.last_days<=35)).groupby('client_id').nd.rank(method='first')
cols=['n','first_days','last_days','med_gap','gap_std','last_dom','amt_med','amt_cv','n_90','next_days','n_ref','last_refunded','rank_next']
W=F.pivot(index='client_id',columns='fam',values=cols); W.columns=[f'{f}_{c}' for c,f in W.columns]
# client-level
g=tx.groupby('client_id')
C=pd.DataFrame({'n_tx':g.size(),'last_any':(CUT-g.timestamp.max()).dt.days,'first_any':(CUT-g.timestamp.min()).dt.days,
  'n_tx_60':tx[tx.timestamp>CUT-pd.Timedelta(days=60)].groupby('client_id').size(),
  'n_salary':tx[tx.description=='salary'].groupby('client_id').size()})
A=F[(F.last_days<=35)&(F.n>=2)]
C['n_active']=A.groupby('client_id').size(); C['min_next']=A.groupby('client_id').next_days.min()
X=C.join(W).reindex(y.index)
yy=y.map({l:i for i,l in enumerate(L)}).values
def cv(X,seed=0):
    oof=np.zeros((len(X),8))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,yy):
        m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=15,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,n_jobs=1,verbose=-1,random_state=seed)
        m.fit(X.iloc[tr],yy[tr]); oof[te]=m.predict_proba(X.iloc[te])
    return oof
oof=cv(X)
p=oof.argmax(1); print('LGBM raw feats argmax macroF1', round(f1_score(yy,p,average='macro'),4))
# tune none threshold scaling
for s in [0.6,0.8,1.0,1.2,1.5]:
    o=oof.copy(); o[:,7]*=s; print(' none x',s, round(f1_score(yy,o.argmax(1),average='macro'),4))
print(pd.Series(f1_score(yy,p,average=None,labels=range(8)),index=L).round(3).to_dict())
m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=15,n_jobs=1,verbose=-1).fit(X,yy)
imp=pd.Series(m.booster_.feature_importance('gain'),index=X.columns).sort_values(ascending=False)
print(imp.head(20).round(0))
