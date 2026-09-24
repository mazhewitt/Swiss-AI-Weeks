import pandas as pd, numpy as np, re
SHOP={'shop','market','foods','fresh','grocery','store','neighborhood','marketplace','hotel','booking','pharmacy','ride','share','coffee','casual','dining','electronics','salary','atm','withdrawal','p2p','send','receive'}
ANCH=[('cloud',{'cloud','storage','backup'}),('gym',{'gym','fit','fitness','club','urban','membership'}),
('insurance',{'cover','insurance','safe','policy'}),('mobile',{'phone','contract','bill'}),
('software',{'saas','productivity','prod','suite','software'}),('streaming',{'media','video'}),('music',{'audio'})]
def anchor(d):
    w=set(d.split())
    if w&SHOP: return 'shop'
    if 'service plan' in d: return 'cloud'
    hits=[f for f,s in ANCH if w&s]
    if len(set(hits))==1: return hits[0]
    if len(set(hits))>1: return 'multi'
    return None
MCCFAM={'5732':'cloud','7997':'gym','6300':'insurance','4814':'mobile','5734':'software','5812':'stream/music'}
def cands(t):
    t=t.copy()
    t['anc']=t.description.map(anchor)
    return t
def cluster(g,tol=0.06):
    g=g.sort_values('amount'); la=np.log(g.amount.values)
    cid=np.concatenate([[0],np.cumsum(np.diff(la)>tol)])
    return pd.Series(cid,index=g.index)
def build(t,tol=0.06):
    t=cands(t)
    c=t[(t.type=='card_payment')&(t.direction=='out')&(t.anc!='shop')&~t.description.isin(['service fee'])].copy()
    c['cl']=c.groupby('client_id',group_keys=False).apply(lambda g:cluster(g,tol))
    return t,c
def summarize(c,cutoff=pd.Timestamp('2026-01-01',tz='UTC')):
    rows=[]
    for (cl,k),g in c.groupby(['client_id','cl']):
        g=g.sort_values('ts'); n=len(g)
        an=g.anc.dropna(); an=an[an!='multi']
        vc=an.value_counts()
        fam=vc.index[0] if len(vc) else None
        purity=vc.iloc[0]/len(an) if len(vc) else np.nan
        gaps=np.diff(g.ts.values).astype('timedelta64[s]').astype(float)/86400 if n>1 else np.array([])
        per=np.median(gaps) if len(gaps) else np.nan
        rows.append(dict(client_id=cl,cl=k,n=n,fam=fam,purity=purity,nanch=len(an),amed=g.amount.median(),acv=g.amount.std()/g.amount.mean() if n>1 else np.nan,
          per=per,gapmad=np.median(np.abs(gaps-per)) if len(gaps) else np.nan,first=g.ts.iloc[0],last=g.ts.iloc[-1],
          mcc=g.mcc.mode().iloc[0],mccfrac=(g.mcc==g.mcc.mode().iloc[0]).mean(),descs=';'.join(g.description.value_counts().index[:6])))
    s=pd.DataFrame(rows)
    s['days_since_last']=(cutoff-s['last']).dt.total_seconds()/86400
    s['next']=s['last']+pd.to_timedelta(s.per,unit='D')
    s['next_d']=(s['next']-cutoff).dt.total_seconds()/86400
    return s

GRP={'5732':'cloud','7997':'gym','6300':'insurance','4814':'mobile','5734':'software','5812':'sm'}
MUSIC_ONLY={'audio','pass'}
def grp_of(r):
    a=None if pd.isna(r.anc) else r.anc
    if a in ('streaming','music'): return 'sm'
    if a and a not in ('multi','shop'): return a
    return GRP.get(r.mcc,'other')
def build2(t,tol=0.06):
    t=cands(t)
    c=t[(t.type=='card_payment')&(t.direction=='out')&(t.anc!='shop')].copy()
    c['grp']=[grp_of(r) for r in c.itertuples()]
    c['sm_hint']=np.where(c.description.str.contains('audio|pass'),'music',np.where(c.description.str.contains('media|video'),'streaming',None))
    c['cl']=c.groupby(['client_id','grp'],group_keys=False).apply(lambda g:cluster(g,tol))
    c['cl']=c.grp+'_'+c.cl.astype(str)
    return t,c
def summarize2(c,cutoff=pd.Timestamp('2026-01-01',tz='UTC')):
    s=summarize(c,cutoff)
    s['grp']=s.cl.str.split('_').str[0]
    h=c.groupby(['client_id','cl']).sm_hint.agg(lambda x:x.dropna().value_counts().index[0] if x.notna().any() else None).rename('smh').reset_index()
    s=s.merge(h,on=['client_id','cl'])
    s['family']=np.where(s.grp=='sm',s.smh.fillna('sm?'),s.grp)
    return s
