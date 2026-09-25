import numpy as np, pandas as pd
import cvrace as cr
d4=pd.read_pickle("cache/race4.pkl").set_index(["client_id","stream_id"])
d=cr.d; key=pd.MultiIndex.from_arrays([d.client_id,d.stream_id])
d["n_fam_refunds"]=d4.n_fam_refunds.reindex(key).to_numpy()
d["unmatched_ref"]=d.n_fam_refunds-d.n_refunded
d["n_refunded"]=d.n_refunded.astype(float)
cr.run(cr.BASE+["n_fam_refunds","unmatched_ref","n_refunded"],tag="+family refunds")
cr.run(cr.BASE+["n_fam_refunds","unmatched_ref","n_refunded","n_ns_sub","n_ns_sub_90"],tag="+family refunds +hidden series")
