"""Which stream-only features drive the LGBM stack? Gain importance, one-feature-at-a-time nonlinear CV AUC
(LGBM on none-prob + that feature), and drop-group ablations. Writes drivers.csv."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import importlib.util, sys
OUT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("stk", OUT / "stack_lib.py"); stk = importlib.util.module_from_spec(spec); spec.loader.exec_module(stk)
L, y, STREAM = stk.L, stk.y, stk.STREAM
base = roc_auc_score(y, L.oof_none)
m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=30, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.7, random_state=0, verbose=-1, n_jobs=1, importance_type="gain").fit(L[["oof_none"] + STREAM].astype(float), y)
imp = pd.Series(m.feature_importances_, index=["oof_none"] + STREAM).sort_values(ascending=False)
print("gain importance top 20:"); print((imp / imp.sum()).head(20).round(3).to_string())
rows = []
for f in STREAM:
    p = stk.oof_pred("lgbm", ["oof_none", f], None)
    rows.append(dict(feature=f, auc_with_none=roc_auc_score(y, p), gain=roc_auc_score(y, p) - base,
                     auc_alone=roc_auc_score(y, stk.oof_pred("lgbm", [f], None))))
R = pd.DataFrame(rows).sort_values("gain", ascending=False)
R.to_csv(OUT / "drivers.csv", index=False)
print(R.head(25).round(4).to_string(index=False))
