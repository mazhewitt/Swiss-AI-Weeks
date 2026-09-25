import pandas as pd, numpy as np
d=pd.read_pickle("cache/race3.pkl"); d["family"]=d.family.astype(str)
lab=pd.read_pickle("cache/lab.pkl").set_index("client_id")["target_next_recurring_merchant"]
P=pd.read_csv("cache/P_base.csv",index_col=0); pred=P.idxmax(axis=1); y=lab.reindex(P.index)
fams=d.groupby("client_id").family.apply(set)
B=[c for c in P.index if y[c]!="none" and pred[c]!="none" and pred[c]!=y[c] and y[c] in fams.get(c,set())]
print("bucket",len(B))
# best stream per family = soonest projected
d["o"]=d._o
first=d.sort_values(["client_id","o","n_payments"],ascending=[True,True,False]).groupby(["client_id","family"]).first().reset_index()
F=["days_to_next","next_rank","n_payments","days_since_first","days_since_last","period_days","gap_mad_days","amount_cv","amount","refund_rate","active","n_desc","n_amt_levels","fee_share","cur_is_client","last_dev","n_mcc","dom_last","hour_med","family_description_share","p_callast","overdue"]
def pairs(clients, a_of, b_of):
    A=first.set_index(["client_id","family"])
    ra=A.loc[[(c,a_of[c]) for c in clients],F].reset_index(drop=True); rb=A.loc[[(c,b_of[c]) for c in clients],F].reset_index(drop=True)
    return ra, rb
ra,rb=pairs(B,y,pred)  # true vs picked
print("\nBUCKET true vs picked: median true, median picked, share true>picked")
for f in F: print(f"{f:26s} {ra[f].median():8.3f} {rb[f].median():8.3f} {np.mean(ra[f]>rb[f]):.2f} (ties {np.mean(ra[f]==rb[f]):.2f})")
print("\ntrue earlier than picked (orig proj):",np.mean(ra.days_to_next<rb.days_to_next).round(3),
      " picked rank1:",np.mean(rb.next_rank==1).round(3))
print("true active:",ra.active.mean().round(3)," picked active:",rb.active.mean().round(3))
# control: correct clients with >=2 families: true vs best other
C=[c for c in P.index if y[c]!="none" and pred[c]==y[c] and len(fams.get(c,set()))>=2]
other={c:P.loc[c,[f for f in fams[c] if f!=y[c]]].astype(float).idxmax() for c in C}
ca,cb=pairs(C,y,other)
print("\nCONTROL (correct) true vs runner-up: share true>other")
for f in ["days_to_next","n_payments","days_since_last","amount_cv","active","n_desc","fee_share","cur_is_client","family_description_share"]:
    print(f"{f:26s} {np.mean(ca[f]>cb[f]):.2f}")
pd.Series(B).to_csv("cache/bucket_base.csv",index=False)
