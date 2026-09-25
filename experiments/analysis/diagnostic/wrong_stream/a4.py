import pandas as pd, numpy as np
c=pd.read_pickle("cache/cand_proj.pkl")
lab=pd.read_pickle("cache/lab.pkl").set_index("client_id")["target_next_recurring_merchant"]
a=c[c.active==1]
nf=a.groupby("client_id").family.nunique()
ok=a[(a.client_id.map(nf)>=2)&(a.y!="none")]
ok=ok[ok.groupby("client_id").family.transform(lambda f:(f==ok.loc[f.index,"y"]).any())]
rng=np.random.default_rng(0); ok=ok.assign(rand=rng.random(len(ok)), rev=-ok.p_orig)
for m in ["p_orig","p_callast","rand","rev"]:
    f=ok.groupby(["client_id","family"])[m].min().reset_index()
    f["rk"]=f.groupby("client_id")[m].rank(method="first"); f["k"]=f.client_id.map(f.groupby("client_id").size())
    f["y"]=f.client_id.map(lab); r=f[f.family==f.y]
    print(m, r.groupby("k").rk.apply(lambda x: f"n={len(x)} " + " ".join(f"{(x==i).mean():.2f}" for i in range(1,5))).to_dict())
# gap between rank1 and label projections when label not rank1
f=ok.groupby(["client_id","family"]).p_orig.min().reset_index().sort_values(["client_id","p_orig"])
f["y"]=f.client_id.map(lab); first=f.groupby("client_id").first()
lp=f[f.family==f.y].set_index("client_id").p_orig
d=(lp-first.p_orig); print("label proj - first proj (when label not first):",d[d>0].describe().round(1).to_dict())
print("share label within 3d of first given not first:",((d>0)&(d<=3)).sum()/(d>0).sum())
