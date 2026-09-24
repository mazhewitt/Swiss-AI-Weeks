"""Task 4: in-history stop hazard of live streams vs the real none rate among live-stream Clients.
Uses the train stream table detected over the whole history (at the Cutoff) with member payments."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from recurring_family.config import CUTOFF
OUT = Path(__file__).parent
D = pickle.load(open(OUT / "prep.pkl", "rb")); C = pickle.load(open(OUT / "client_feats.pkl", "rb"))
H = pickle.load(open(OUT / "hidden_client.pkl", "rb"))
st, members, lab = D["streams"], D["members"], D["labels"]
DAY = pd.Timedelta(days=1)
per = st.set_index(["client_id", "stream_id"]).period_days
res = []
lines = []
for c in pd.date_range("2025-03-01", "2025-10-01", freq="MS", tz="UTC").append(pd.DatetimeIndex([pd.Timestamp("2025-10-03", tz="UTC")])):
    live_streams = []; client_live = {}; client_pay = {}
    for m in members:
        t = m["times"]; before = t[t < c]
        if len(before) < 3: continue
        gaps = np.asarray((before[1:] - before[:-1]) / DAY, float)
        P = float(np.median(gaps / np.maximum(1, np.round(gaps / np.median(gaps)))))
        if (c - before[-1]) / DAY > 1.6 * P: continue
        paid = bool(((t >= c) & (t < c + 90 * DAY)).any())
        live_streams.append(paid)
        client_live[m["client_id"]] = client_live.get(m["client_id"], 0) + 1
        client_pay[m["client_id"]] = client_pay.get(m["client_id"], False) or paid
    # any stream payment (any stream with >= 2 payments overall) in the Horizon, for live Clients
    any_pay = {}
    for m in members:
        t = m["times"]
        if len(t) >= 2 and ((t >= c) & (t < c + 90 * DAY)).any(): any_pay[m["client_id"]] = True
    lc = list(client_live)
    full = c + 90 * DAY <= CUTOFF
    res.append(dict(shifted_cutoff=c.date(), horizon_fully_observed=full, live_streams=len(live_streams),
                    stream_stop_90d=1 - np.mean(live_streams), live_clients=len(lc),
                    all_live_stop_90d=np.mean([not client_pay[x] for x in lc]),
                    live_client_none_90d=np.mean([not any_pay.get(x, False) for x in lc])))
R = pd.DataFrame(res)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
R.to_csv(OUT / "stop_hazard.csv", index=False)
# per-payment hazard over the full history: after an established stream's k-th payment (k>=3), does it stop?
stops = opp = 0
for r in st.itertuples():
    if r.n_payments < 3: continue
    stopped = (not r.active)
    opp += (r.n_payments - 3) + (1 if stopped else 0)
    stops += 1 if stopped else 0
h = stops / opp
print(f"per-payment stop hazard {h:.4f} ({stops} stops / {opp} opportunities); per 90 days (monthly) {1-(1-h)**(90/30.4):.3f}")
# real none rate among live-stream Clients, overall and by number of live streams; with vs without hidden marker
L = C[C.n_live > 0].join(H[["h_n2"]])
print("real none rate | live:", round(L.none.mean(), 3), " | live & no hidden series:", round(L[L.h_n2 == 0].none.mean(), 3))
print(L.groupby(L.n_live.clip(upper=4)).none.mean().round(3).to_dict(), L[L.h_n2 == 0].groupby(L.n_live.clip(upper=4)).none.mean().round(3).to_dict())
pickle.dump(dict(h=h, stops=stops, opp=opp), open(OUT / "hazard.pkl", "wb"))
