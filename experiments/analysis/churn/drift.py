"""Label-free drift check on the cached stream tables (train / valid / unlabeled at the Cutoff): do the
stream-only churn features shift? Only stream caches are read; no valid label, no valid transactions."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from recurring_family.streams import _digest, StreamParams
from recurring_family.config import CUTOFF, SHIFTED_CUTOFF
key = _digest((str(CUTOFF), StreamParams())); skey = _digest((str(SHIFTED_CUTOFF), StreamParams()))
print("cutoff key", key, "shifted key", skey)
tabs = {}
for p in Path("artifacts/streams").glob("*.pkl"):
    split = p.name.split("-")[0]
    if p.name.endswith(key + ".pkl"): tabs[split] = pd.read_pickle(p)
print({k: len(v) for k, v in tabs.items()})
rows = []
for split, s in tabs.items():
    live = s[s.active]
    per_client = s.groupby("client_id")
    lc = live.groupby("client_id")
    rows.append(dict(split=split, clients=s.client_id.nunique(), streams_per_client=len(s) / s.client_id.nunique(),
                     live_per_client=len(live) / s.client_id.nunique(), share_clients_live=lc.ngroups / s.client_id.nunique(),
                     live_fam_desc_share_mean=live.family_description_share.mean(),
                     live_fam_desc_share_lt_0_67=(live.family_description_share < 0.67).mean(),
                     live_amount_cv_median=live.amount_cv.median(), live_amount_cv_gt_0_02=(live.amount_cv > 0.02).mean(),
                     live_refund_rate_mean=live.refund_rate.mean(), live_n_payments_median=live.n_payments.median(),
                     client_min_amount_cv_median=lc.amount_cv.min().median(),
                     client_max_fam_desc_share_mean=lc.family_description_share.max().mean(),
                     filler_top_description=(s.description.isin(["member plan", "subscription charge", "digital service"])).mean(),
                     short_streams_per_client=(s.n_payments < 3).sum() / s.client_id.nunique()))
R = pd.DataFrame(rows).set_index("split").T
print(R.round(4).to_string()); R.to_csv(Path(__file__).parent / "drift_streams.csv")
