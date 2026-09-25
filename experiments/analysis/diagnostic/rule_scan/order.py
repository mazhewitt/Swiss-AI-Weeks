import pandas as pd, numpy as np
pd.set_option('display.width',250)
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
F=pd.read_pickle(S+'F2.pkl'); y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
F['y']=F.client_id.map(y)
A=F[(F.last_days<=35)&(F.n>=2)].copy()
nn=A[A.y!='none']; ok=nn.groupby('client_id').is_label.any(); nn=nn[nn.client_id.isin(ok[ok].index)]
print('non-none clients w/ label among active:',nn.client_id.nunique(),'of',(y!='none').sum())
A['next_dom']=A.last_dom; A['neg_n']=-A.n; A['neg_amt']=-A.amt_med; A['neg_first']=-A.first_days
for col in ['next_days','last_days','next_dom','first_days','neg_first','n','neg_n','amt_med','neg_amt','amt_cv','gap_std']:
    g=nn.assign(**{c:A.loc[nn.index,c] for c in ['next_dom','neg_n','neg_amt','neg_first']}) if col in['next_dom','neg_n','neg_amt','neg_first'] else nn
    top=g.sort_values(col).groupby('client_id').first()
    print(f'{col:12s} acc={top.is_label.mean():.3f}')
# fam prior among ties
print('label fam share, non-none:',y[y!='none'].value_counts(normalize=True).round(3).to_dict())
# none clients: how many active streams do they have?
na=A.groupby('client_id').size(); 
d=pd.DataFrame({'y':y}); d['n_act']=d.index.map(na).fillna(0)
print(pd.crosstab(d.n_act, d.y=='none'))
# for rows with label, next_days distribution of the label stream vs others
print(nn.groupby('is_label').next_days.describe().round(1))
print(nn.groupby('is_label')[['last_days','n','amt_cv','gap_std','first_days']].median())
