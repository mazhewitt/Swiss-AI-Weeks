"""How reliable is the family of an unhinted (no audio/video word) 5812 stream? Label==own vs label==sibling."""
from lib import *
_, y, st = load()
st["y"] = st.client_id.map(y)
sib = {"music": "streaming", "streaming": "music"}
fams = st.groupby("client_id")["family"].agg(set)
ms = st[st.family.isin(["music", "streaming"])].copy()
ms["sibling_absent"] = [sib[f] not in fams[c] for c, f in zip(ms.client_id, ms.family)]
ms["own"] = ms.y == ms.family; ms["sib"] = ms.y == ms.family.map(sib)
ms["hinted"] = np.where(ms.family_description_share >= 0.999, "all", np.where(ms.family_description_share > 0, "some", "none"))
ms["short"] = np.where(ms.n_payments <= 2, "n<=2", "n>=3")
ms["amt_band"] = pd.cut(ms.median_amount, [0, 12, 15.5, 19, 1000])
g = ms[ms.sibling_absent].groupby(["hinted", "short"])[["own", "sib"]].agg(["sum", "size"])
print(g)
g = ms[ms.sibling_absent & (ms.hinted == "none")].groupby(["amt_band"])[["own", "sib"]].sum()
print(g)
