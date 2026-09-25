"""What the schedule join adds, per split (label-free), and on train which Clients get joins (train labels)."""
from pathlib import Path
import numpy as np
import pandas as pd
from recurring_family import data, streams as st
from recurring_family.config import CUTOFF

log = []
orig = st._join_on_schedule


def spy(stray, stream_of, ranges, days, log_amount, params):
    before = stream_of.copy()
    orig(stray, stream_of, ranges, days, log_amount, params)
    for i in stray:
        if stream_of[i] >= 0:
            times = days[before == stream_of[i]]
            log.append("gap" if times.min() < days[i] < times.max() else "extends")


st._join_on_schedule = spy
rows = []
for split in ("train", "valid", "test", "unlabeled"):
    tx = data.load_transactions(Path("data/raw"), split)
    seen = len(log)
    for client, pay, srows, membership, _ in st._detect_per_client(tx, CUTOFF, st.StreamParams()):
        mine = log[seen:]
        seen = len(log)
        rows.append({"split": split, "client_id": client, "gap": mine.count("gap"), "extends": mine.count("extends"),
                     "n_payments": int((membership >= 0).sum())})
    print(split, flush=True)
df = pd.DataFrame(rows)
df["joins"] = df.gap + df.extends
print(df.groupby("split")[["gap", "extends", "joins", "n_payments"]].mean().round(3))
print("share of Clients with a join:", df.groupby("split").joins.apply(lambda j: (j > 0).mean()).round(3).to_dict())
with data.training_run():
    labels = data.load_labels(Path("data/raw"), "train").set_index("client_id")["target_next_recurring_merchant"]
t = df[df.split == "train"].set_index("client_id")
t["none"] = labels.reindex(t.index) == "none"
print("train none rate: all", round(t.none.mean(), 3), "| Clients with a join", round(t[t.joins > 0].none.mean(), 3),
      "| joins per Client, none vs other", t.groupby("none").joins.mean().round(3).to_dict())
