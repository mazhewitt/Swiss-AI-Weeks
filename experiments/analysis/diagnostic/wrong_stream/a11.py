import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from recurring_family.ranker import FEATURE_COLUMNS, LGBM_PARAMS
lab=pd.read_pickle("cache/lab.pkl").set_index("client_id")["target_next_recurring_merchant"]
B=pd.read_csv("cache/bucket_base.csv").iloc[:,0]
for f in ["P_base","P_both"]:
    P=pd.read_csv(f"cache/{f}.csv",index_col=0); pr=P.idxmax(axis=1)
    print(f, "bucket acc", (pr[B]==lab[B]).mean().round(3), " bucket->none", (pr[B]=="none").mean().round(3))
# survival AUC among active rows, CV, base vs extended
d=pd.read_pickle("cache/race3.pkl"); d["famcode"]=pd.Categorical(d.family.astype(str)).codes
folds=pd.read_csv("../../../../artifacts/hailmary/rehearsal/oof_surv.csv",index_col=0)["fold"]; d["fold"]=d.client_id.map(folds)
BASE=[f if f!="family" else "famcode" for f in FEATURE_COLUMNS]
EXT=BASE+["n_desc","n_amt_levels","fee_share","cur_is_client","last_dev","n_mcc","dom_last","hour_med","cents","last_gap_ratio","max_gap","n_refunded","wd_last"]
r=d[d.t.notna()].copy()
for name,F in [("base",BASE),("ext",EXT),("ext+ns",EXT+["n_ns_sub","n_ns_sub_90","n_tx","salary","n_refund","fee_rate","last_tx_days","n_card"])]:
    r["s"]=np.nan
    for k in range(5):
        tr=r[r.fold!=k]; m=lgb.LGBMClassifier(**LGBM_PARAMS).fit(tr[F],tr.t); r.loc[r.fold==k,"s"]=m.predict_proba(r.loc[r.fold==k,F])[:,1]
    a=r[r.active==1]
    # within-client: pairs of active streams in non-none clients, one with t=1 & one t=0
    print(name,"AUC all",round(roc_auc_score(r.t,r.s),3),"active",round(roc_auc_score(a.t,a.s),3),
          "active non-none clients",round(roc_auc_score(a[a.y!='none'].t,a[a.y!='none'].s),3))
    if name=="ext+ns":
        imp=pd.Series(m.feature_importances_,index=F).sort_values(ascending=False); print(imp.head(15).to_dict())
