"""Cache train streams, member payments, candidates, labels and OOF preds (train only)."""
from pathlib import Path
import pandas as pd
from recurring_family.data import load_transactions, load_labels
from recurring_family.ranker import _detected, candidates

OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True)
tx = load_transactions(Path("data/raw"), "train")
lab = load_labels(Path("data/raw"), "train")
print(type(lab), getattr(lab, "columns", None))
clients = pd.Index(tx["client_id"].astype(str).unique())
streams, paid = _detected(tx, clients)
cand = candidates(streams)
cand["stream_id"] = streams["stream_id"].to_numpy()
for c in ("mcc", "description", "family_description_share", "next_payment", "last_payment", "first_payment"):
    cand[c] = streams[c].to_numpy()
tx.to_pickle(OUT / "tx.pkl"); streams.to_pickle(OUT / "streams.pkl"); paid.to_pickle(OUT / "paid.pkl")
cand.to_pickle(OUT / "cand.pkl"); pd.to_pickle(lab, OUT / "lab.pkl")
print(len(tx), len(streams), len(paid), len(clients))
