"""(client, family) pairs where the family has NO detected stream: which pre-cutoff evidence predicts label==family?"""
from lib import *
from recurring_family.config import CUTOFF
tx = pd.read_pickle(OUT / "tx.pkl"); _, y, st = load()
fams = st.groupby("client_id")["family"].agg(set)
DAY = pd.Timedelta(days=1)
free = tx[tx.stream_family.isna() & tx.rawfam.notna() & (tx.type != "fee")].copy()
# expand music_or_streaming to both
free = pd.concat([free[free.rawfam != "music_or_streaming"],
                  free[free.rawfam == "music_or_streaming"].assign(rawfam="music"),
                  free[free.rawfam == "music_or_streaming"].assign(rawfam="streaming")])
free["cat"] = np.select([
    (free.type == "card_payment") & (free.direction == "out") & free.decoy,
    (free.type == "card_payment") & (free.direction == "out"),
    free.type == "refund"], ["pay_decoy", "pay", "refund"], "other_" + free.type.astype(str))
rows = []
for c in y.index:
    have = fams.get(c, set())
    g = free[free.client_id == c]
    for F in FAMS:
        if F in have: continue
        h = g[g.rawfam == F]
        r = dict(client_id=c, fam=F, y=int(y[c] == F), ynone=int(y[c] == "none"))
        for k in ("pay", "pay_decoy", "refund"):
            hk = h[h.cat == k]
            r["n_" + k] = len(hk)
            r["last_" + k] = (CUTOFF - hk.timestamp.max()) / DAY if len(hk) else np.nan
        r["n_other"] = int(h.cat.str.startswith("other").sum())
        r["n_det_fams"] = len(have)
        rows.append(r)
P = pd.DataFrame(rows); P.to_pickle(OUT / "pairs.pkl")
print("undetected-family pairs", len(P), "positives", P.y.sum(), "base rate", P.y.mean().round(4))
def tab(mask, name):
    s = P[mask]; print(f"{name:45s} pairs={len(s):6d} pos={s.y.sum():4d} prec={s.y.mean():.3f} share_of_undet={s.y.sum()/P.y.sum():.3f}")
tab(P.index >= 0, "all")
for k in ("pay", "pay_decoy", "refund"):
    tab(P["n_" + k] >= 1, f"{k}>=1")
    tab(P["n_" + k] == 0, f"{k}==0")
    tab((P["n_" + k] >= 1) & (P["last_" + k] <= 30), f"{k} in last 30d")
    tab((P["n_" + k] >= 1) & (P["last_" + k] <= 14), f"{k} in last 14d")
    tab(P["n_" + k] >= 2, f"{k}>=2")
tab((P.n_pay + P.n_pay_decoy + P.n_refund) == 0, "no evidence at all")
tab(P.n_det_fams == 0, "client has no stream at all")
print(P[P.y == 1].n_det_fams.value_counts().sort_index())
print(P.groupby("n_det_fams").y.agg(["mean", "sum", "size"]))
