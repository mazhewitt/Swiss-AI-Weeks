import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
F=pd.read_pickle(S+'F.pkl'); c=pd.read_pickle(S+'tagged.pkl'); tx=pd.read_pickle(S+'tx.pkl')
y=pd.read_pickle(S+'lb.pkl').set_index('client_id').target_next_recurring_merchant.astype(str)
anyf=F.groupby('client_id').fam.agg(set)
d=pd.DataFrame({'y':y}).join(anyf); d['fam']=d.fam.apply(lambda s:s if isinstance(s,set) else set())
miss=d[(d.y!='none')&~d.apply(lambda r:r.y in r.fam,axis=1)]
print('non-none label with ZERO tagged payments of that family:',len(miss), miss.y.value_counts().to_dict())
for cl in miss.index[:5]:
    print('==',cl,'label',miss.y[cl],'fams',miss.fam[cl])
    g=tx[(tx.client_id==cl)&(tx.type.isin(['card_payment','refund']))]
    g=g[~g.description.isin(['fresh foods','pharmacy','hotel booking','electronics shop','ride share','coffee shop','grocery store','neighborhood market','online marketplace','casual dining'])]
    print(g[['timestamp','type','amount','currency','mcc','description','fee']].tail(25).to_string())
