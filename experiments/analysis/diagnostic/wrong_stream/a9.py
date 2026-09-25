import pandas as pd, numpy as np
d=pd.read_pickle("cache/race3.pkl"); d["family"]=d.family.astype(str)
lab=pd.read_pickle("cache/lab.pkl").set_index("client_id")["target_next_recurring_merchant"]
B=set(pd.read_csv("cache/bucket_base.csv").iloc[:,0])
first=d.sort_values(["client_id","_o","n_payments"],ascending=[True,True,False]).groupby(["client_id","family"]).first().reset_index()
first["y"]=first.client_id.map(lab); first["is_label"]=first.family==first.y
tb=first[first.client_id.isin(B)&first.is_label]
print("bucket true stream: n_payments dist", tb.n_payments.clip(upper=6).value_counts().sort_index().to_dict())
print("bucket true stream active", tb.active.mean().round(3))
print("bucket true inactive: days_since_last quantiles", tb[tb.active==0].days_since_last.quantile([.1,.25,.5,.75,.9]).round(0).to_dict())
# Label rate of non-active streams by n_payments & days_since_last (all clients)
na=first[first.active==0].copy()
na["dsl"]=pd.cut(na.days_since_last,[0,15,31,45,62,90,400])
print("\nnon-active stream: P(label==family) by n_payments x days_since_last")
print(na.pivot_table(index=na.n_payments.clip(upper=4),columns="dsl",values="is_label",aggfunc=["mean","size"]).round(2).to_string())
ac=first[first.active==1]
print("\nactive: P(label==family) by next_rank",ac.groupby(ac.next_rank.clip(upper=4)).is_label.agg(["mean","size"]).round(3).to_dict())
