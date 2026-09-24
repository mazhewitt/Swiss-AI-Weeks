"""Orphan structure: do orphans form hidden amount clusters (streams the detector misses)? Timing by label.
Also compare unlabeled (0 decoys) orphan rates, and the valid stream cache vs train on filler-sensitive columns."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from recurring_family.streams import _evidence
from recurring_family.config import CUTOFF
OUT = Path(__file__).parent
D = pickle.load(open(OUT / "prep.pkl", "rb")); C = pickle.load(open(OUT / "client_feats.pkl", "rb"))
tx, st, members, lab = D["tx"], D["streams"], D["members"], D["labels"]
in_stream = set(np.concatenate([m["idx"] for m in members]))
cp = tx[tx.type == "card_payment"].copy()
cp["kind"] = [_evidence(d, m)[0] for d, m in zip(cp.description, cp.mcc)]
o = cp[(cp.kind != "drop") & ~cp.index.isin(in_stream)].copy()
o["y"] = o.client_id.map(lab)
# hidden clusters: per client, cluster orphan log-amounts with tol 0.06, count clusters with >=3 members
def clusters(g):
    la = np.sort(np.log(g.amount.to_numpy()))
    br = np.concatenate([[0], np.cumsum(np.diff(la) > 0.06)])
    sizes = np.bincount(br)
    return pd.Series({"n_orphan": len(la), "max_cluster": sizes.max(), "n_clusters3": (sizes >= 3).sum(),
                      "first_orphan_days": (CUTOFF - g.timestamp.min()).days})
H = o.groupby("client_id").apply(clusters)
H = H.reindex(C.index).fillna({"n_orphan": 0, "max_cluster": 0, "n_clusters3": 0})
H["none"] = C.none; H["live"] = C.n_live > 0
print(H.groupby("none")[["n_orphan", "max_cluster", "n_clusters3", "first_orphan_days"]].mean())
print("share with any orphan by label:", H.groupby("none").n_orphan.apply(lambda s: (s > 0).mean()).to_dict())
print("P(none) by orphan count (live clients):")
Lh = H[H.live]
print(Lh.groupby(Lh.n_orphan.clip(upper=8)).none.agg(["size", "mean"]).round(3).T)
print("AUC orphan-count (live)", roc_auc_score(Lh.none, Lh.n_orphan), "max_cluster", roc_auc_score(Lh.none, Lh.max_cluster))
# orphans by label per 30-day bucket, per Client of each label
n_by = C.groupby("none").size()
o["mb"] = (CUTOFF - o.timestamp).dt.days // 30
prof = o.groupby(["mb", o.y.eq("none")]).size().unstack().fillna(0) / n_by.values
print("orphans per Client per 30 days before Cutoff (col True = none):"); print(prof.head(14).round(3).T)
# stream payments per client per month by label for scale
