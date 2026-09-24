import pandas as pd, numpy as np
from pathlib import Path
from recurring_family import data
from recurring_family.streams import cached_streams
RAW=Path("data/raw")
with data.training_run():
    labels=data.load_labels(RAW,"train").set_index("client_id")["target_next_recurring_merchant"]
    tx=data.load_transactions(RAW,"train")
print(labels.value_counts())
print(tx.shape, tx.timestamp.min(), tx.timestamp.max())
print(tx.type.value_counts()); print(tx.groupby(["type","direction"]).size())
s,_=cached_streams(RAW,"train",Path("artifacts/streams"))
print(s.shape); print(s.head().T)
print(s.groupby("client_id").active.any().mean())
