from lib import *
tx, y, st = load()
tx.to_pickle(OUT / "tx.pkl")
fams = st.groupby("client_id")["family"].agg(set)
undet = [c for c in y.index if y[c] != "none" and y[c] not in fams.get(c, set())]
pd.set_option("display.width", 250, "display.max_rows", 3000, "display.max_colwidth", 40)
cols = ["timestamp", "type", "direction", "amount", "currency", "mcc", "description", "fee", "kind", "rawfam", "stream_family"]
for c in undet[:25]:
    g = tx[(tx.client_id == c) & tx.rawfam.notna() & (tx.type != "fee")]
    print(f"\n=== {c} label={y[c]} streams={sorted(fams.get(c, set()))}")
    print(g[g.stream_family.isna()][cols].to_string())
