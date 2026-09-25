from pathlib import Path
import pandas as pd
from recurring_family.data import load_transactions, load_labels
pd.set_option('display.width',200); pd.set_option('display.max_rows',300)
S=Path('/tmp/claude-0/-home-user-Swiss-AI-Weeks/a2bd0492-5644-507f-9faf-1e938cbf2568/scratchpad')
tx=load_transactions(Path('data/raw'),'train'); lb=load_labels(Path('data/raw'),'train')
tx.to_pickle(S/'tx.pkl'); lb.to_pickle(S/'lb.pkl')
print(tx.shape, tx.client_id.nunique(), lb.shape); print(tx.dtypes)
print(tx.timestamp.min(), tx.timestamp.max())
print(lb.target_next_recurring_merchant.value_counts())
for c in ['type','direction','currency']: print(tx[c].value_counts())
print(tx.fee.describe()); print(tx[tx.fee>0].type.value_counts())
print(tx.mcc.value_counts().head(60))
print(tx.description.value_counts().head(120))
print(tx.description.nunique(), tx.mcc.nunique())
