"""Odd payments by months before cutoff and by type of oddness, none vs family label (all stream members)."""
from lib import *
from recurring_family.config import CUTOFF
m = pd.read_pickle(OUT / "members.pkl"); _, y, st = load()
DAY = pd.Timedelta(days=1)
m["mb"] = ((CUTOFF - m.timestamp) / DAY // 30.44).astype(int)
m["w"] = m.description.str.lower().str.split()
BASE = {"media streaming","phone contract","cloud access","cover plan","member pass","urban gym","saas billing","productivity suite",
 "service bill","gym membership","fit club","video access","cloud backup","storage plan","service plan","audio streaming",
 "software access","insurance monthly","fitness monthly","safe cover","policy premium","digital plus","premium plan","monthly plan"}
FILL = {"member plan", "subscription charge", "digital service"}
d = m.description.str.lower()
m["otype"] = np.select([m.off_home, d.isin(FILL), d.str.split().str.len() == 1, d.str.contains(r"^(pay|billing|member) "), d.str.contains(r" (core|online|plus|digital|service)$") & ~d.isin(BASE), ~d.isin(BASE)],
                       ["off_home", "filler", "fragment", "prefix", "suffix", "other_variant"], "base")
pd.set_option("display.width", 250)
print(m.groupby([m.mb.clip(upper=12), "none"]).odd.mean().unstack().round(3).T)
print(pd.crosstab(m[m.mb <= 4].otype, m[m.mb <= 4].none, normalize="columns").round(3))
print(pd.crosstab(m[m.mb >= 7].otype, m[m.mb >= 7].none, normalize="columns").round(3))
print(m[(m.mb <= 4) & (m.otype == "other_variant")].description.value_counts().head(15))
print(m.groupby([m.mb.clip(upper=12), "none"]).odd.mean().unstack().round(3).to_string())
