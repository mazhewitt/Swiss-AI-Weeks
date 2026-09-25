"""Where does the word "subscription" (and its neighbours) live? Per split: the share of Clients with such a
card payment, how many, how recent, on which MCC, in a detected stream or stray; in train split by label.
Train labels only; other splits' transactions only."""
import re
from pathlib import Path
import numpy as np, pandas as pd
from recurring_family import data
from recurring_family.cli import Paths, _training_data
from recurring_family.config import CUTOFF
from recurring_family.ranker import _detected
from recurring_family.streams import _canonical_description

DAY = pd.Timedelta(days=1)
paths = Paths(Path("."))
WORDS = ["subscription", "member", "plan", "service", "charge", "renewal", "monthly", "premium", "digital", "order", "merchant", "payment", "purchase"]

def enrich(tx):
    card = tx[tx["type"] == "card_payment"].copy()
    card["days"] = (CUTOFF - card["timestamp"]) / DAY
    card["canon"] = card["description"].map(_canonical_description)
    streams, paid = _detected(tx, pd.Index(sorted(tx["client_id"].unique())))
    pkey = set(paid["client_id"].astype(str) + "|" + paid["timestamp"].astype(str) + "|" + paid["amount"].round(2).astype(str))
    card["member"] = (card["client_id"].astype(str) + "|" + card["timestamp"].astype(str) + "|" + card["amount"].round(2).astype(str)).isin(pkey)
    return card

with data.training_run():
    tx, labels = _training_data(paths, with_selection=False)
cards = {"train": enrich(tx)}
for split in ("valid", "test", "unlabeled"):
    cards[split] = enrich(data.load_transactions(paths.raw, split))
n_clients = {s: c["client_id"].nunique() for s, c in cards.items()}
is_none = (labels == "none")

print("=== share of Clients with a card payment whose description contains the word ===")
rows = []
for w in WORDS:
    row = {"word": w}
    for split, c in cards.items():
        has = c[c["description"].str.contains(rf"\b{w}\b", case=False, regex=True)].groupby("client_id").size()
        if split == "train":
            row["train none"] = round(has.reindex(labels.index[is_none]).notna().mean(), 3)
            row["train family"] = round(has.reindex(labels.index[~is_none]).notna().mean(), 3)
        else:
            row[split] = round(len(has) / n_clients[split], 3)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print('\n=== "subscription" payments: canonical descriptions, per split (payments per 100 Clients) ===')
sub = {s: c[c["description"].str.contains(r"\bsubscription\b", case=False)] for s, c in cards.items()}
tab = pd.DataFrame({s: d["canon"].value_counts() / n_clients[s] * 100 for s, d in sub.items()}).fillna(0).round(1)
print(tab.sort_values("train", ascending=False).head(15).to_string())

print('\n=== "subscription" payments in train by label: count per Client, recency, MCC, in a stream? ===')
t = sub["train"].assign(none=sub["train"]["client_id"].map(is_none))
for flag, name in ((True, "none Clients"), (False, "family Clients")):
    d = t[t["none"] == flag]
    per = d.groupby("client_id").agg(n=("days", "size"), last_days=("days", "min"), first_days=("days", "max"))
    print(f"{name}: {len(per)} Clients with any; payments per such Client {per['n'].mean():.2f}; last one {per['last_days'].median():.0f} days before the Cutoff (median), "
          f"q10-q90 {per['last_days'].quantile(.1):.0f}-{per['last_days'].quantile(.9):.0f}; in a detected stream {d['member'].mean():.2f}; MCC {d['mcc'].value_counts(normalize=True).round(2).head(6).to_dict()}")
print("\nper split, all 'subscription' payments: per Client with any, recency, stream membership")
for s, d in sub.items():
    per = d.groupby("client_id").agg(n=("days", "size"), last_days=("days", "min"))
    print(f"  {s:9s}: {len(per)/n_clients[s]:.3f} of Clients; payments per such Client {per['n'].mean():.2f}; last one median {per['last_days'].median():.0f} days, q10-q90 {per['last_days'].quantile(.1):.0f}-{per['last_days'].quantile(.9):.0f}; in a stream {d['member'].mean():.2f}")

print("\n=== train: none share by number of 'subscription' payments, and by recency of the last one ===")
per = sub["train"].groupby("client_id").agg(n=("days", "size"), last_days=("days", "min")).reindex(labels.index).fillna({"n": 0})
per["none"] = is_none.to_numpy()
print(per.groupby(per["n"].clip(upper=4))["none"].agg(["size", "mean"]).round(3).to_string())
per["recency"] = pd.cut(per["last_days"], [0, 15, 30, 60, 120, 1000])
print(per[per["n"] > 0].groupby("recency", observed=True)["none"].agg(["size", "mean"]).round(3).to_string())

print("\n=== descriptions most skewed to none in train (Client share none vs family), with their test share ===")
rows = []
for canon, g in cards["train"].groupby("canon"):
    has = g.groupby("client_id").size().index
    sn, sf = is_none.reindex(has).sum() / is_none.sum(), (~is_none).reindex(has).sum() / (~is_none).sum()
    if sn + sf > 0.05:
        rows.append({"description": canon, "train none": round(sn, 3), "train family": round(sf, 3),
                     **{s: round(cards[s][cards[s]["canon"] == canon]["client_id"].nunique() / n_clients[s], 3) for s in ("valid", "test", "unlabeled")}})
d = pd.DataFrame(rows); d["ratio"] = (d["train none"] + .01) / (d["train family"] + .01)
print(d.sort_values("ratio", ascending=False).head(12).to_string(index=False))
print(d.sort_values("ratio").head(6).to_string(index=False))
