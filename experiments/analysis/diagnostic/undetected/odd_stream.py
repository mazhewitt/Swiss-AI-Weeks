"""Within family-labelled Clients: are the label stream's recent payments less odd than the other streams'?"""
from lib import *
from recurring_family.config import CUTOFF
from sklearn.metrics import roc_auc_score
m = pd.read_pickle(OUT / "members.pkl"); _, y, st = load()
DAY = pd.Timedelta(days=1)
m["mb"] = ((CUTOFF - m.timestamp) / DAY // 30.44).astype(int)
r = m[m.mb <= 5].groupby(["client_id", "stream_id"]).agg(rec_odd=("odd", "mean"), rec_n=("odd", "size")).reset_index()
s = st.merge(r, on=["client_id", "stream_id"], how="left")
s["y"] = s.client_id.map(y); s["is_label"] = s.family == s.y
fam = s[(s.y != "none") & s.active & s.rec_n.notna()]
multi = fam.groupby("client_id").filter(lambda g: len(g) >= 2 and g.is_label.any())
print("active streams in multi-stream family-labelled clients", len(multi))
print(multi.groupby("is_label").rec_odd.describe().round(3))
print("AUC(label vs -rec_odd)", round(roc_auc_score(multi.is_label, -multi.rec_odd), 3))
print(multi.groupby(pd.cut(multi.rec_odd, [-0.01, 0, 0.2, 0.4, 1.0])).is_label.agg(["mean", "size"]).round(3))
# per-stream survival target in the none-free sense: all streams in none clients get odd; within-client stream-level?
nn = s[(s.y == "none") & s.active & s.rec_n.notna()]
print("none clients: active-stream rec_odd", nn.rec_odd.describe().round(3).to_dict())
# is oddness client-wide in none clients? share of none clients where ALL active streams have rec_odd>0
g = nn.groupby("client_id").rec_odd.agg(lambda v: (v > 0).mean())
print("none clients with >=2 active streams: mean share of their streams with any recent odd", g[nn.groupby('client_id').size() >= 2].mean().round(3))
