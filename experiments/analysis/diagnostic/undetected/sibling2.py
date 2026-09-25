from lib import *
from recurring_family.config import CUTOFF
_, y, st = load()
fams = st.groupby("client_id")["family"].agg(set)
sib = {"music": "streaming", "streaming": "music"}
pd.set_option("display.width", 250, "display.max_rows", 300)
cols = ["client_id", "family", "period_days", "median_amount", "n_payments", "last_payment", "active", "mcc", "description", "family_description_share"]
st["y"] = st.client_id.map(y)
ms = st[st.family.isin(["music", "streaming"])].copy()
ms["sib_label"] = [(yy == sib[f]) and (sib[f] not in fams[c]) for c, f, yy in zip(ms.client_id, ms.family, ms.y)]
ms["own_label"] = ms.y == ms.family
print(ms[ms.sib_label][cols + ["y"]].to_string())
# amount distribution of music vs streaming streams where label is own family
print(ms[ms.own_label].groupby("family").median_amount.describe())
print(ms[ms.sib_label].groupby("family").median_amount.describe())
print("unhinted share among sib-label streams", (ms[ms.sib_label].family_description_share < 1).mean(), "vs own", (ms[ms.own_label].family_description_share < 1).mean())
