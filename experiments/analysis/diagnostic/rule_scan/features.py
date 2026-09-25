"""Raw per-client, per-family features without the pipeline's stream detector."""
import numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
FAMS=['cloud','gym','insurance','mobile','music','software','streaming']
CUT=pd.Timestamp('2026-01-01',tz='UTC')
HOME={'4814':'mobile','6300':'insurance','7997':'gym','5732':'cloud','5734':'software'}
KW=[('music',r'audio|member pass'),('streaming',r'media stream|video'),('mobile',r'phone|service bill'),
    ('insurance',r'cover|policy|insurance|safe'),('gym',r'gym|fit|urban'),('cloud',r'cloud|storage|service plan'),
    ('software',r'saas|productivity|prod suite|suite|software')]
EVERYDAY={'salary','atm withdrawal','fresh foods','pharmacy','hotel booking','electronics shop','ride share','coffee shop',
 'grocery store','neighborhood market','online marketplace','casual dining','p2p send','p2p receive','service fee'}
DECOY={'digital order','merchant charge','service payment','card purchase'}
def tag(tx):
    c=tx[(tx.type=='card_payment')&~tx.description.isin(EVERYDAY)].copy()
    c['decoy']=c.description.isin(DECOY)
    c['fam']=None; s=c.description.str
    for f,p in KW: c.loc[c.fam.isna()&s.contains(p)&~c.decoy,'fam']=f
    c['kw']=c.fam.notna()
    # ambiguous: home mcc
    amb=c.fam.isna()&~c.decoy
    c.loc[amb,'fam']=c.loc[amb,'mcc'].map(HOME)
    # remaining ambiguous: nearest known-family amount for that client (within 8%)
    known=c[c.kw].groupby(['client_id','fam']).amount.median().reset_index()
    rest=c[c.fam.isna()&~c.decoy]
    out={}
    kd={k:g for k,g in known.groupby('client_id')}
    for i,r in rest.iterrows():
        g=kd.get(r.client_id)
        if g is None: continue
        if r.mcc=='5812': g=g[g.fam.isin(['music','streaming'])]
        if not len(g): continue
        rel=(g.amount-r.amount).abs()/g.amount
        j=rel.idxmin()
        if rel[j]<0.08: out[i]=g.fam[j]
    c.loc[list(out),'fam']=pd.Series(out)
    return c
def feats(c):
    c=c[c.fam.notna()].sort_values('timestamp')
    rows=[]
    for (cl,f),g in c.groupby(['client_id','fam']):
        t=g.timestamp; gaps=t.diff().dt.total_seconds().div(86400).dropna()
        last=t.iloc[-1]; mg=gaps.median() if len(gaps) else np.nan
        rows.append(dict(client_id=cl,fam=f,n=len(g),n_kw=int(g.kw.sum()),first_days=(CUT-t.iloc[0]).days,
          last_days=(CUT-last).total_seconds()/86400, med_gap=mg, gap_std=gaps.std() if len(gaps)>1 else np.nan,
          last_dom=last.day, amt_med=g.amount.median(), amt_cv=g.amount.std()/g.amount.mean() if len(g)>1 else np.nan,
          n_90=int((t>CUT-pd.Timedelta(days=90)).sum()), n_last_mcc_home=int(g.mcc.isin(HOME).sum()),
          next_days=(mg-(CUT-last).total_seconds()/86400) if len(gaps) else np.nan))
    return pd.DataFrame(rows)
if __name__=='__main__':
    tx=pd.read_pickle(S+'tx.pkl'); c=tag(tx); c.to_pickle(S+'tagged.pkl')
    print('tagged',c.fam.notna().sum(),'untagged non-decoy',(c.fam.isna()&~c.decoy).sum(),'decoy',c.decoy.sum())
    print(c[c.fam.isna()&~c.decoy].description.value_counts().head(15))
    F=feats(c); F.to_pickle(S+'F.pkl'); print(F.describe().T)
