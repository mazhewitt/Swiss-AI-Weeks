"""Train only: is `none` / 'not the soonest' consistent with each Active Stream independently stopping at the Cutoff?"""
from pathlib import Path
import numpy as np, pandas as pd
from recurring_family import data, streams as st
from recurring_family.ranker import candidates
RAW = Path("data/raw")
lab = data.load_labels(RAW, "train").set_index("client_id")[data.LABEL_COLUMN]
s, _ = st.cached_streams(RAW, "train", Path("artifacts/streams"))
c = candidates(s).assign(family=lambda d: d.family.astype(str))
a = c[c.active == 1].sort_values("days_to_next").groupby(["client_id", "family"], as_index=False).first()
a["order"] = a.sort_values("days_to_next").groupby("client_id").cumcount() + 1
k = a.groupby("client_id").size()
rows = []
for kk in sorted(k.unique()):
    ids = k[k == kk].index
    t = lab.reindex(ids)
    pos = [ (a[(a.client_id == i) & (a.family == t[i])]["order"].iloc[0] if (t[i] != "none" and ((a.client_id == i) & (a.family == t[i])).any()) else (0 if t[i] == "none" else -1)) for i in ids]
    pos = pd.Series(pos)
    rows.append(dict(k=kk, n=len(ids), none=(pos == 0).mean(), first=(pos == 1).mean(), second=(pos == 2).mean(),
                     third_plus=(pos >= 3).mean(), not_active=(pos == -1).mean()))
r = pd.DataFrame(rows)
print(r.round(3).to_string(index=False))
# fit one survival s from none rate across k: none = (1-s)^k
m = r[r.n >= 30]
for _, x in m.iterrows():
    s1 = 1 - x.none ** (1 / x.k)
    exp_first = s1  # P(soonest survives)
    print(f"k={int(x.k)}: implied s={s1:.2f}; if independent, P(label=1st)={exp_first:.2f}, 2nd={(1-s1)*s1:.2f}  observed 1st={x.first:.2f} 2nd={x.second:.2f}")
print("\nClients with no active stream:", int((~lab.index.isin(k.index)).sum()), "none rate:", round((lab[~lab.index.isin(k.index)] == "none").mean(), 3))
