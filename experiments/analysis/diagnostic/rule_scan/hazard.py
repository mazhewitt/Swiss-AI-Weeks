import pandas as pd, numpy as np
pd.set_option('display.width',250)
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
F=pd.read_pickle(S+'F2.pkl'); c=pd.read_pickle(S+'tagged.pkl'); c=c[c.fam.notna()].sort_values('timestamp')
y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
CUT=pd.Timestamp('2026-01-01',tz='UTC')
# extra per-stream: last gap/median gap, amount trend, currency count, last desc kw?, last mcc home?
def ex(g):
    t=g.timestamp; gaps=t.diff().dt.days.dropna()
    return pd.Series(dict(last_gap_ratio=(gaps.iloc[-1]/gaps.median()) if len(gaps) else np.nan,
      amt_trend=g.amount.iloc[-3:].mean()/g.amount.iloc[:3].mean() if len(g)>=4 else np.nan,
      n_cur=g.currency.nunique(), last_kw=g.kw.iloc[-1], last_home=g.mcc.iloc[-1] in {'4814','6300','7997','5732','5734','5812'},
      last_fee=g.fee.iloc[-1]>0, last_hour=t.iloc[-1].hour, n_desc=g.description.nunique()))
E=c.groupby(['client_id','fam']).apply(ex)
F=F.set_index(['client_id','fam']).join(E).reset_index()
F['y']=F.client_id.map(y)
A=F[(F.last_days<=35)&(F.n>=2)].copy()
first=A.sort_values('next_days').groupby('client_id').head(1).copy()
first['hit']=first.is_label; first['ynone']=first.y=='none'
print('soonest active stream: P(label)=',first.hit.mean().round(3),' P(none)=',first.ynone.mean().round(3), 'N',len(first))
def show(col,bins=None):
    k=pd.qcut(first[col],5,duplicates='drop') if bins is None else pd.cut(first[col],bins)
    print(first.groupby(k,observed=True)[['hit','ynone']].mean().round(3).assign(n=first.groupby(k,observed=True).size()))
for col in ['n','first_days','last_days','next_days','amt_cv','gap_std','last_gap_ratio','amt_trend','amt_med']: print('--',col); show(col)
for col in ['n_cur','last_kw','last_home','last_fee','n_desc','fam']: print('--',col); print(first.groupby(col)[['hit','ynone']].mean().round(3).assign(n=first.groupby(col).size()))
F.to_pickle(S+'F3.pkl')
