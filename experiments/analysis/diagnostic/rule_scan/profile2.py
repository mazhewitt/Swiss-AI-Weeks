import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',300); pd.set_option('display.max_columns',30)
S='/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad/'
tx=pd.read_pickle(S+'tx.pkl')
print(pd.crosstab(tx.type,tx.mcc))
print(pd.crosstab(tx.type,tx.direction))
everyday={'salary','atm withdrawal','fresh foods','pharmacy','hotel booking','electronics shop','ride share','coffee shop','grocery store','neighborhood market','online marketplace','casual dining','p2p send','p2p receive','service fee'}
sub=tx[~tx.description.isin(everyday)]
print(sub.type.value_counts(), sub.direction.value_counts())
top=sub.description.value_counts().head(35).index
print(pd.crosstab(sub[sub.description.isin(top)].description, sub[sub.description.isin(top)].mcc))
print(pd.crosstab(tx[tx.description.isin(everyday)].description, tx[tx.description.isin(everyday)].mcc))
print(sub.groupby('description').amount.describe().loc[top])
print(tx[tx.type=='refund'].description.value_counts().head(20))
print(tx.timestamp.dt.hour.value_counts().sort_index().values)
