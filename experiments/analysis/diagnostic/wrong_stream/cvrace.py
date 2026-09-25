"""5-fold CV Survival Race (argmax, untuned) with feature/order variants. Train only."""
import pandas as pd, numpy as np, lightgbm as lgb, sys, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from recurring_family.ranker import FEATURE_COLUMNS, LGBM_PARAMS
L=["cloud","gym","insurance","mobile","music","software","streaming","none"]
d=pd.read_pickle("cache/race3.pkl"); d["family"]=d.family.astype(str)
lab=pd.read_pickle("cache/lab.pkl").set_index("client_id")["target_next_recurring_merchant"]
folds=pd.read_csv("../../../../artifacts/hailmary/rehearsal/oof_surv.csv",index_col=0)["fold"]
d["famcode"]=pd.Categorical(d.family,categories=L[:-1]).codes
BASE=[f if f!="family" else "famcode" for f in FEATURE_COLUMNS]
def targets(g, order):
    g=g.sort_values([order,"n_payments"],ascending=[True,False]); y=g.y.iat[0]; t=pd.Series(np.nan,index=g.index)
    if y=="none": t[:]=0
    else:
        hit=np.flatnonzero(g.family.to_numpy()==y)
        if len(hit): t.iloc[:hit[0]]=0; t.iloc[hit[0]]=1
    return t
def run(feats, order="_o", tag=""):
    d["ord"]=d[order].fillna(np.inf) if order!="_o" else d["_o"]
    d["tt"]=d.groupby("client_id",group_keys=False).apply(lambda g:targets(g,"ord"))
    d["fold"]=d.client_id.map(folds); d["s"]=np.nan
    for k in range(5):
        tr=d[(d.fold!=k)&d.tt.notna()]; te=d.fold==k
        m=lgb.LGBMClassifier(**LGBM_PARAMS).fit(tr[feats],tr.tt,categorical_feature=["famcode"])
        d.loc[te,"s"]=m.predict_proba(d.loc[te,feats])[:,1]
    P=pd.DataFrame(0.0,index=lab.index,columns=L); P["none"]=1.0
    for cid,g in d.sort_values(["client_id","ord","n_payments"],ascending=[True,True,False]).groupby("client_id"):
        surv=1.0; row=dict.fromkeys(L,0.0)
        for f,s in zip(g.family,g.s): row[f]+=surv*s; surv*=1-s
        row["none"]=surv; P.loc[cid]=pd.Series(row)
    pred=P.idxmax(axis=1); y=lab.reindex(P.index)
    print(f"{tag:40s} mF1 {f1_score(y,pred,average='macro'):.4f} acc {(pred==y).mean():.4f}")
    return P
if __name__=="__main__":
    P0=run(BASE,tag="base 16 feats, orig order"); P0.to_csv("cache/P_base.csv")
    run(BASE,"p_callast",tag="base, calendar-last-DOM order")
    run(BASE,"p_reg",tag="base, regression order")
    X1=BASE+["n_ns_sub","n_ns_sub_90"]; run(X1,tag="+hidden series (client)")
    X2=BASE+["n_desc","n_amt_levels","fee_share","cur_is_client","last_dev","n_mcc","dom_last"]; P2=run(X2,tag="+stream payment feats")
    X3=X1+X2[len(BASE):]; P3=run(X3,tag="+both"); P3.to_csv("cache/P_both.csv")
