"""Undetected labels: sibling-family streams (music<->streaming), and the pre-cutoff state of clients with no stream."""
from lib import *
_, y, st = load()
fams = st.groupby("client_id")["family"].agg(set)
und = pd.Series({c: y[c] for c in y.index if y[c] != "none" and y[c] not in fams.get(c, set())})
sib = {"music": "streaming", "streaming": "music"}
ms = und[und.isin(["music", "streaming"])]
print("music/streaming undetected", len(ms), "with sibling stream", sum(sib[f] in fams.get(c, set()) for c, f in ms.items()))
nost = [c for c in y.index if c not in fams.index]
print("clients with no stream", len(nost), y[nost].value_counts().to_dict())
# label distribution of undetected vs all family labels
print(pd.concat([und.value_counts(), y[y != "none"].value_counts()], axis=1, keys=["undet", "all"]).assign(rate=lambda d: d.undet / d["all"]))
