import pandas as pd
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
tx=pd.read_pickle(S+'tx.pkl'); c=tx[tx.type=='card_payment']
def base(d): return d
m=c.description.str
c=c.assign(fam=None)
kw=[('music',r'audio|member pass'),('streaming',r'media stream|video'),('mobile',r'phone|service bill'),('insurance',r'cover|policy|insurance|safe'),('gym',r'gym|fit'),('cloud',r'cloud|storage|service plan'),('software',r'saas|productivity|prod suite|software')]
for f,p in kw: c.loc[c.fam.isna()&m.contains(p),'fam']=f
for d in ['digital plus','premium plan']:
  x=c[c.description==d]
  for cl,g in list(x[x.mcc=='5812'].groupby('client_id'))[:6]:
    y=c[(c.client_id==cl)&c.fam.isin(['music','streaming'])]
    print(d,cl,g.amount.round(2).tolist()[:5],g.timestamp.dt.day.tolist()[:5],'| known:',y.groupby('fam').amount.median().round(2).to_dict())
# how many clients have 5812 digital plus without any keyword music/streaming?
x=c[(c.mcc=='5812')&c.description.isin(['digital plus','premium plan'])]
kn=c[c.fam.isin(['music','streaming'])].groupby('client_id').fam.agg(set)
print(x.groupby('client_id').description.first().map(lambda d:d).to_frame().join(kn).assign(k=lambda d:d.fam.astype(str)).groupby(['description','k']).size())
