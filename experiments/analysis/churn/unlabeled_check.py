"""Unlabeled split (no labels, 0 decoys): stream detection at the Cutoff with membership (not cached to the repo),
then the same filler-sensitive stream columns and the orphan / hidden-series rates, to compare with train."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from recurring_family import data
from recurring_family.config import CUTOFF
from recurring_family.streams import _detect_per_client, StreamParams, _evidence
OUT = Path(__file__).parent
def profile(tx, name):
    rows, orphan_rows = [], []
    for client, pays, srows, membership, _ in _detect_per_client(tx, CUTOFF, StreamParams()):
        rows.extend(srows)
        kinds = np.array([_evidence(d, m)[0] for d, m in zip(pays.description, pays.mcc)])
        o = pays[(kinds != "drop") & (membership < 0)]
        n_or = len(o); mx = 0; n2 = 0
        if n_or:
            la = np.sort(np.log(o.amount.to_numpy())); br = np.concatenate([[0], np.cumsum(np.diff(la) > 0.06)])
            sz = np.bincount(br); mx = sz.max(); n2 = (sz >= 2).sum()
        orphan_rows.append(dict(client_id=client, n_orphan=n_or, hidden_max_n=mx, hidden_n2=n2))
    s = pd.DataFrame(rows); O = pd.DataFrame(orphan_rows)
    live = s[s.active]; lc = live.groupby("client_id")
    out = dict(split=name, clients=s.client_id.nunique(), live_per_client=len(live) / s.client_id.nunique(),
               live_fam_desc_share_mean=live.family_description_share.mean(),
               live_fam_desc_share_lt_0_67=(live.family_description_share < 0.67).mean(),
               live_amount_cv_gt_0_02=(live.amount_cv > 0.02).mean(), live_refund_rate_mean=live.refund_rate.mean(),
               orphans_per_client=O.n_orphan.mean(), share_clients_hidden_n2=(O.hidden_n2 > 0).mean(),
               share_clients_hidden_max_n_ge3=(O.hidden_max_n >= 3).mean())
    return out, s, O
with data.training_run():
    tr = data.load_transactions(Path("data/raw"), "train")
un = data.load_transactions(Path("data/raw"), "unlabeled")
a, _, Otr = profile(tr, "train"); b, su, Oun = profile(un, "unlabeled")
R = pd.DataFrame([a, b]).set_index("split").T
print(R.round(4).to_string()); R.to_csv(OUT / "drift_unlabeled.csv")
pickle.dump(dict(streams=su, orphans=Oun), open(OUT / "unlabeled_profile.pkl", "wb"))
