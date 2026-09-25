"""Leak checks on train (labels allowed): does the label correlate with anything that is not a signal?
Client id order, label-file row order, history length and end, transaction counts; and do any split's
transactions extend past the Cutoff?"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from recurring_family import data
from recurring_family.cli import Paths, _training_data
from recurring_family.config import CUTOFF

paths = Paths(Path("."))
with data.training_run():
    tx, labels = _training_data(paths, with_selection=False)
raw_labels = pd.read_csv(paths.raw / "train_labels.csv", dtype=str)
is_none = (labels == "none").astype(int)
print(f"train Clients {len(labels)}, none share {is_none.mean():.3f}")

def auc(name, x):
    x = pd.Series(x).reindex(labels.index)
    ok = x.notna()
    a = roc_auc_score(is_none[ok], x[ok])
    print(f"  {name:40s} AUC for none {max(a, 1-a):.3f} ({'high' if a >= 0.5 else 'low'} -> none)")

print("\nid and file order")
auc("numeric client id", labels.index.str[1:].astype(int).to_series(index=labels.index))
auc("row in label file", pd.Series(np.arange(len(raw_labels)), index=raw_labels["client_id"]))
none_in_order = (raw_labels["target_next_recurring_merchant"] == "none").astype(int).to_numpy()
lag1 = np.corrcoef(none_in_order[:-1], none_in_order[1:])[0, 1]
print(f"  lag-1 autocorrelation of none in file order {lag1:+.3f} (independent: ~0 +- {1/np.sqrt(len(none_in_order)):.3f})")
fam = raw_labels["target_next_recurring_merchant"]
runs = (fam != fam.shift()).sum()
print(f"  label runs in file order {runs} of {len(fam)} rows (independent expectation ~{int(len(fam) * (1 - (fam.value_counts(normalize=True)**2).sum()))})")

print("\nhistory shape")
g = tx.groupby("client_id")
per = pd.DataFrame({
    "n_tx": g.size(), "n_card": g.apply(lambda d: (d["type"] == "card_payment").sum()),
    "first_ts_days": (CUTOFF - g["timestamp"].min()) / pd.Timedelta(days=1),
    "last_ts_days": (CUTOFF - g["timestamp"].max()) / pd.Timedelta(days=1),
    "last_card_days": g.apply(lambda d: (CUTOFF - d.loc[d["type"] == "card_payment", "timestamp"].max()) / pd.Timedelta(days=1)),
    "n_types": g["type"].nunique(), "n_desc": g["description"].nunique(), "n_mcc": g["mcc"].nunique(),
    "sum_in": g.apply(lambda d: d.loc[d["direction"] == "in", "amount"].sum()),
    "sum_out": g.apply(lambda d: d.loc[d["direction"] == "out", "amount"].sum()),
    "n_refund_like": g.apply(lambda d: d["type"].str.contains("refund", case=False).sum()),
})
for c in per: auc(c, per[c])
print(f"  any train transaction after the Cutoff: {(tx['timestamp'] >= CUTOFF).sum()}")
print("  transaction types:", tx["type"].value_counts().to_dict())
print("  directions:", tx["direction"].value_counts().to_dict(), " currencies:", tx["currency"].value_counts().to_dict())
for split in ("valid", "test", "unlabeled"):
    t = data.load_transactions(paths.raw, split)
    print(f"  {split}: {t['client_id'].nunique()} Clients, max timestamp {t['timestamp'].max()}, after Cutoff {(t['timestamp'] >= CUTOFF).sum()}")
