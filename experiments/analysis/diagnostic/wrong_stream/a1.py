import pandas as pd, numpy as np
from sklearn.metrics import f1_score
C="cache/"
cand=pd.read_pickle(C+"cand.pkl"); lab=pd.read_pickle(C+"lab.pkl").set_index("client_id")["target_next_recurring_merchant"]
L=["cloud","gym","insurance","mobile","music","software","streaming","none"]
for name in ["v2","surv"]:
    o=pd.read_csv(f"../../../../artifacts/hailmary/rehearsal/oof_{name}.csv",index_col=0)
    pred=o[L].idxmax(axis=1); y=lab.reindex(o.index)
    fams=cand.groupby("client_id")["family"].apply(lambda s:set(s.astype(str)))
    has=[(yy in fams.get(c,set())) for c,yy in zip(o.index,y)]
    has=pd.Series(has,index=o.index)
    wrong=(pred!=y)
    b=has&wrong&(pred!="none")
    print(name,"argmax mF1",round(f1_score(y,pred,average="macro"),4),"acc",round((pred==y).mean(),3),
      "bucket",b.sum(),"of",len(y), "| label none",(y=="none").sum(),"has-family",has.sum())
    # oracle: fix bucket
    p2=pred.copy(); p2[b]=y[b]; print("  fix bucket ->",round(f1_score(y,p2,average="macro"),4))
    pd.Series(b[b].index).to_csv(f"cache/bucket_{name}.csv",index=False)
    print(pd.crosstab(y[b],pred[b]))
