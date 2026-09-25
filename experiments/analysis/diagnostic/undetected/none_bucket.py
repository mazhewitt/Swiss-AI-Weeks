"""Truth-none Clients with Active Streams: observable last-payment signals (train only)."""
from lib import *
from recurring_family.config import CUTOFF
from recurring_family.streams import HOME_MCC
tx = pd.read_pickle(OUT / "tx.pkl"); _, y, st = load()
DAY = pd.Timedelta(days=1)
act = st[st.active].copy()
m = tx[tx.stream_id.notna()].copy(); m["stream_id"] = m.stream_id.astype(int)
m = m.merge(act[["client_id", "stream_id", "family", "period_days", "median_amount", "last_payment", "next_payment"]], on=["client_id", "stream_id"])
m = m.sort_values(["client_id", "stream_id", "timestamp"])
refunds = tx[(tx.type == "refund")]
feats = []
for (c, s), g in m.groupby(["client_id", "stream_id"]):
    last = g.iloc[-1]; prev = g.iloc[:-1]
    words = str(last.description).lower().split()
    home = {k for k, v in HOME_MCC.items() if v == last.family or (v == "music_or_streaming" and last.family in ("music", "streaming"))}
    r = refunds[(refunds.client_id == c) & (refunds.timestamp >= last.timestamp) & (refunds.timestamp <= last.timestamp + 7 * DAY)
                & (np.abs(np.log(refunds.amount / last.amount)) < 0.06)]
    feats.append(dict(client_id=c, stream_id=s,
        last_refunded=len(r) > 0,
        last_ratio=last.amount / prev.amount.median() if len(prev) else np.nan,
        last_ccy_change=len(prev) > 0 and last.currency != prev.currency.mode().iloc[0],
        n_ccy=g.currency.nunique(),
        last_off_home=str(last.mcc) not in home,
        last_fee=last.fee > 0,
        last_generic=not any(w in set().union(*[set(v) for v in []]) for w in words),
        last_suffix=words[-1] if words[-1] in ("core", "online", "plus", "digital", "service") and len(words) > 1 else "-",
        last_prefix=words[0] if words[0] in ("pay", "billing", "member") and len(words) > 1 else "-",
        overdue=((CUTOFF - last.timestamp) / DAY) / last.period_days,
        last_gap_ratio=((last.timestamp - prev.timestamp.iloc[-1]) / DAY) / last.period_days if len(prev) else np.nan,
        amount_trend=g.amount.iloc[-3:].mean() / g.amount.iloc[:3].mean(),
        family=last.family))
F = pd.DataFrame(feats); F["none"] = F.client_id.map(y).eq("none")
F.to_pickle(OUT / "none_stream_feats.pkl")
print("active streams", len(F), "clients", F.client_id.nunique(), "none rate (stream-level)", F.none.mean().round(3))
for col in ["last_refunded", "last_ccy_change", "last_off_home", "last_fee", "last_suffix", "last_prefix", "n_ccy", "family"]:
    print(F.groupby(col).none.agg(["mean", "size"]).round(3).T.to_string(), "\n")
for col in ["last_ratio", "overdue", "last_gap_ratio", "amount_trend"]:
    print(col, F.groupby(pd.qcut(F[col], 5, duplicates="drop")).none.agg(["mean", "size"]).round(3).T.to_string(), "\n")
# client level: all active streams share property
C = F.groupby("client_id").agg(none=("none", "first"), n=("none", "size"), any_ref=("last_refunded", "any"), max_over=("overdue", "max"), min_over=("overdue", "min"))
print("client-level none rate", C.none.mean().round(3))
print(C.groupby("n").none.agg(["mean", "size"]).round(3).T)
print(C.groupby(pd.qcut(C.min_over, 5)).none.agg(["mean", "size"]).round(3).T)
