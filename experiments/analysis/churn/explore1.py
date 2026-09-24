import pickle, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from recurring_family.config import CUTOFF
D=pickle.load(open(Path(__file__).parent/"prep.pkl","rb"))
lab,st,oof,tx=D["labels"],D["streams"],D["oof"],D["tx"]
live=st[st.active].groupby("client_id").size()
cl=pd.DataFrame({"y":lab}); cl["none"]=(cl.y=="none").astype(int)
cl["n_live"]=live.reindex(cl.index).fillna(0)
cl["n_streams"]=st.groupby("client_id").size().reindex(cl.index).fillna(0)
cl["pnone"]=oof["none"]
cl["pmax"]=oof[[c for c in oof.columns if c not in("fold","none")]].max(axis=1)
print(cl.groupby(cl.n_live.clip(upper=4)).agg(n=("none","size"),none=("none","mean")))
L=cl[cl.n_live>0]
print("live clients",len(L),"none rate",L.none.mean())
print("AUC pnone",roc_auc_score(L.none,L.pnone),"AUC 1-pmax",roc_auc_score(L.none,1-L.pmax))
print("AUC all clients",roc_auc_score(cl.none,cl.pnone))
# the truth-family: is the true family among the live stream families?
fam_live=st[st.active].groupby("client_id").family.apply(set)
Lf=L[L.none==0]
print("truth family in live set", np.mean([y in fam_live[c] for c,y in Lf.y.items()]))
fam_all=st.groupby("client_id").family.apply(set)
F=cl[cl.none==0]; print("truth family in any stream", np.mean([c in fam_all and y in fam_all[c] for c,y in F.y.items()]))
# soonest next payment among live
nx=st[st.active].copy(); nx["dn"]=(nx.next_payment-CUTOFF).dt.days
first=nx.sort_values("dn").groupby("client_id").first()
print("truth == soonest live family", np.mean([first.family[c]==y for c,y in Lf.y.items()]))
