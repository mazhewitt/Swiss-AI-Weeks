"""Is the 'odd' payment (off-home MCC or variant description) a last-payment effect or stream-wide? Train only."""
from lib import *
from recurring_family.config import CUTOFF
from recurring_family.streams import HOME_MCC
tx = pd.read_pickle(OUT / "tx.pkl"); _, y, st = load()
BASE = {"media streaming","phone contract","cloud access","cover plan","member pass","urban gym","saas billing","productivity suite",
 "service bill","gym membership","fit club","video access","cloud backup","storage plan","service plan","audio streaming",
 "software access","insurance monthly","fitness monthly","safe cover","policy premium","digital plus","premium plan","monthly plan"}
m = tx[tx.stream_id.notna()].copy(); m["stream_id"] = m.stream_id.astype(int)
m = m.merge(st[["client_id","stream_id","family","active","n_payments"]], on=["client_id","stream_id"])
def home_ok(mcc, fam):
    h = HOME_MCC.get(str(mcc)); return h == fam or (h == "music_or_streaming" and fam in ("music","streaming"))
m["off_home"] = [not home_ok(a, b) for a, b in zip(m.mcc, m.family)]
m["variant"] = ~m.description.str.lower().isin(BASE)
m["odd"] = m.off_home | m.variant
m = m.sort_values(["client_id","stream_id","timestamp"])
m["pos_from_end"] = m.groupby(["client_id","stream_id"]).cumcount(ascending=False)
m["none"] = m.client_id.map(y).eq("none")
a = m[m.active]
print("odd rate by position-from-end (active streams), none vs not:")
print(a.assign(p=a.pos_from_end.clip(upper=6)).groupby(["p","none"]).odd.mean().unstack().round(3))
print(a.assign(p=a.pos_from_end.clip(upper=6)).groupby(["p","none"]).off_home.mean().unstack().round(3))
# client level rules
last = a[a.pos_from_end == 0]
cl = last.groupby("client_id").agg(none=("none","first"), any_odd=("odd","any"), any_off=("off_home","any"), all_odd=("odd","all"), n=("odd","size"))
print("clients with active streams", len(cl), "none rate", cl.none.mean().round(3))
for col in ["any_off","any_odd","all_odd"]:
    print(col, cl.groupby(col).none.agg(["mean","size","sum"]).round(3).to_dict("index"))
# last 2 payments both odd
l2 = a[a.pos_from_end <= 1].groupby(["client_id","stream_id"]).odd.all().rename("last2_odd").reset_index()
l2["none"] = l2.client_id.map(y).eq("none")
print("stream: last2 odd", l2.groupby("last2_odd").none.agg(["mean","size"]).round(3).to_dict("index"))
cl.to_pickle(OUT / "odd_clients.pkl"); m.to_pickle(OUT / "members.pkl")
