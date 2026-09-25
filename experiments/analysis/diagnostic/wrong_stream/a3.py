import pandas as pd, numpy as np
cand=pd.read_pickle("cache/cand.pkl"); P=pd.read_pickle("cache/proj.pkl")
lab=pd.read_pickle("cache/lab.pkl").set_index("client_id")["target_next_recurring_merchant"]
c=cand.merge(P,on=["client_id","stream_id"]); c["family"]=c.family.astype(str)
c["p_orig"]=c.days_to_next
c["y"]=c.client_id.map(lab)
methods=["p_orig","p_med","p_last","p_mean","p_3044","p_reg","p_cal","p_callast","neg_last"]
c["neg_last"]=c["last_d"]  # the latest last payment first? (proxy) - actually earliest-last first
c.to_pickle("cache/cand_proj.pkl")
for pool in ["active","all_multi"]:
    a=c[c.active==1] if pool=="active" else c[c.n_payments>=2]
    nf=a.groupby("client_id").family.nunique()
    ok=a[a.client_id.map(nf)>=2]; ok=ok[ok.y!="none"]
    ok=ok[ok.groupby("client_id").family.transform(lambda f:(f==ok.loc[f.index,"y"]).any())]
    print(f"\n== pool {pool}: clients {ok.client_id.nunique()}")
    for m in methods:
        f=ok.groupby(["client_id","family"])[m].min().reset_index()
        f["rk"]=f.groupby("client_id")[m].rank(method="first")
        f["y"]=f.client_id.map(lab); r=f[f.family==f.y].rk
        # also rank-1 margin
        print(f"{m:10s} label rank1 {np.mean(r==1):.3f} rank2 {np.mean(r==2):.3f} mean {r.mean():.2f}")
