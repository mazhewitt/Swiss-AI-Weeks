"""How new is the label's stream when detected? (n_payments, first payment) label vs non-label streams."""
from lib import *
from recurring_family.config import CUTOFF
_, y, st = load()
DAY = pd.Timedelta(days=1)
st["y"] = st.client_id.map(y); st["is_label"] = st.family == st.y
st["age"] = (CUTOFF - st.first_payment) / DAY
st["nb"] = pd.cut(st.n_payments, [0, 1, 2, 3, 5, 1000])
st["ab"] = pd.cut(st.age, [0, 35, 65, 120, 1000])
print(pd.crosstab(st.nb, st.is_label, normalize="columns").round(3))
print(pd.crosstab(st.ab, st.is_label, normalize="columns").round(3))
print("P(label) by n_payments band:\n", st.groupby("nb").is_label.agg(["mean", "size"]).round(3))
print("P(label) by age band:\n", st.groupby("ab").is_label.agg(["mean", "size"]).round(3))
