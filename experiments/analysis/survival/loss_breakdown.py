"""Where does v2 lose macro-F1 on the selection set? Oracle-fix each error bucket and rescore."""
from pathlib import Path
import pandas as pd
from recurring_family import data, streams as st
from recurring_family.ranker import candidates
from recurring_family.evaluation import score

RAW = Path("data/raw")
pred = pd.read_csv("experiments/runs/20260925T001755-e309a3.csv", dtype=str).set_index("client_id")["predicted"]
with data.scoring():
    lab = data.load_labels(RAW, "selection").set_index("client_id")[data.LABEL_COLUMN]
truth = lab.reindex(pred.index)
s, _ = st.cached_streams(RAW, "valid", Path("artifacts/streams"))
c = candidates(s)
c = c[c.client_id.isin(pred.index)].assign(family=lambda d: d.family.astype(str))
fams = c.groupby("client_id")["family"].agg(lambda x: set(map(str, x)))
act = c[c.active == 1].groupby("client_id")["family"].agg(lambda x: set(map(str, x)))
fams = fams.reindex(pred.index).apply(lambda v: v if isinstance(v, set) else set())
act = act.reindex(pred.index).apply(lambda v: v if isinstance(v, set) else set())

def bucket(i):
    t, p = truth[i], pred[i]
    if t == p: return "correct"
    if t == "none": return "1 none -> family (churn missed)"
    has = t in fams[i]
    if p == "none":
        return "2 family -> none, true family detected" if has else "3 family -> none, true family NOT detected"
    return "4 wrong family, true family detected" if has else "5 wrong family, true family NOT detected"

b = pd.Series({i: bucket(i) for i in pred.index})
base = score(truth, pred)["macro_f1"]
print(f"selection n={len(pred)} macro-F1 {base:.4f}\n")
rows = []
for name in sorted(b.unique()):
    if name == "correct": continue
    fixed = pred.copy(); m = b == name; fixed[m] = truth[m]
    rows.append((name, int(m.sum()), score(truth, fixed)["macro_f1"] - base))
print(pd.DataFrame(rows, columns=["bucket", "clients", "F1 if fixed"]).to_string(index=False))

# finer: within detected-but-wrong, is the true family's stream active? is it the soonest?
print("\nTrue family detected but missed (buckets 2+4): true-family stream active?")
m = b.str.startswith(("2", "4"))
print(pd.Series([truth[i] in act[i] for i in b[m].index]).value_counts().to_string())
print("\nbucket 1 (truth none): predicted family had an ACTIVE stream?")
m = b.str.startswith("1")
print(pd.Series([pred[i] in act[i] for i in b[m].index]).value_counts().to_string())
print("\nper-label error counts by bucket:")
print(pd.crosstab(truth[b != "correct"], b[b != "correct"]).to_string())
print("\nClients with no candidate stream at all:", int((fams.apply(len) == 0).sum()),
      "of which truth none:", int(((fams.apply(len) == 0) & (truth == "none")).sum()))

print("\n=== bucket 4 detail ===")
m4 = b.str.startswith("4")
best = c.sort_values("days_to_next").groupby(["client_id", "family"]).first().reset_index()
r = []
for i in b[m4].index:
    ci = best[best.client_id == i].set_index("family")
    t, p = ci.loc[truth[i]], ci.loc[pred[i]]
    r.append(dict(n_fams=len(ci), t_rank=t.next_rank, p_rank=p.next_rank, t_days=t.days_to_next, p_days=p.days_to_next,
                  t_active=t.active, p_active=p.active, t_n=t.n_payments, p_n=p.n_payments, t_period=t.period_days, p_period=p.period_days))
r = pd.DataFrame(r)
print("families detected per Client:", r.n_fams.value_counts().sort_index().to_dict())
print("true family is soonest (rank 1):", int((r.t_rank == 1).sum()), " predicted is soonest:", int((r.p_rank == 1).sum()))
print("true active / pred active:", int(r.t_active.sum()), int(r.p_active.sum()))
d = (r.t_days - r.p_days)
print("true days_to_next minus predicted's (quantiles):", d.quantile([.1,.25,.5,.75,.9]).round(1).to_dict(), " NaN:", int(d.isna().sum()))
print("|gap| <= 3 days:", int((d.abs() <= 3).sum()), " <= 7:", int((d.abs() <= 7).sum()))
print("true period quantiles:", r.t_period.quantile([.1,.5,.9]).round(0).to_dict(), " pred period:", r.p_period.quantile([.1,.5,.9]).round(0).to_dict())
# correct Clients for contrast
mc = (b == "correct") & (truth != "none")
k = sum(1 for i in b[mc].index if (best.client_id == i).sum() > 1)
print("correct family Clients with >1 detected family:", k, "of", int(mc.sum()))

print("\n=== family-naming view ===")
m = b.str.startswith(("3", "5"))
print("true family NOT detected (buckets 3+5): truth x predicted")
print(pd.crosstab(truth[m], pred[m]).to_string())
print("\nof those, does the Client have ANY stream (so a misnamed stream could be the true one)?",
      pd.Series([len(fams[i]) > 0 for i in b[m].index]).value_counts().to_dict())
