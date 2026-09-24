import pandas as pd
S='/private/tmp/claude-501/-Users-mazhewitt-projects-Swiss-AI-Weeks/c72de104-7192-40c6-9be6-56eefe922a3c/scratchpad/'
for n in ['train','valid','test']:
    df=pd.read_json(S+f'{n}_transactions.jsonl',lines=True,dtype={'mcc':str})
    df['ts']=pd.to_datetime(df.timestamp)
    df.to_parquet(f'{n}.parquet')
    print(n,df.shape,df.client_id.nunique())
df=pd.read_json(S+'unlabeled_pretrain_transactions.jsonl',lines=True,dtype={'mcc':str})
df['ts']=pd.to_datetime(df.timestamp); df.to_parquet('unl.parquet'); print('unl',df.shape,df.client_id.nunique())
