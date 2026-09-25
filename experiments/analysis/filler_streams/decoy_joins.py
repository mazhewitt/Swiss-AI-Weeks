"""Label-free: Decoy-described payments the detector puts in a stream, per 100 streams of 2+ payments, per split."""
from pathlib import Path
import numpy as np
import pandas as pd
from recurring_family import data
from recurring_family.config import CUTOFF
from recurring_family.streams import StreamParams, _decoy_described, _detect_per_client

rows = []
for split in ("train", "valid", "test", "unlabeled"):
    tx = data.load_transactions(Path("data/raw"), split)
    n_streams = joined = inside = clients = 0
    for client, pay, srows, membership, _ in _detect_per_client(tx, CUTOFF, StreamParams()):
        n_streams += sum(r["n_payments"] >= 2 for r in srows)
        t = pd.DatetimeIndex(pay["timestamp"]).asi8
        decoy = np.array([_decoy_described(d) for d in pay["description"]])
        hit = np.flatnonzero(decoy & (membership >= 0))
        joined += len(hit)
        clients += bool(len(hit))
        for i in hit:
            others = np.flatnonzero((membership == membership[i]) & ~decoy)
            inside += bool(len(others)) and t[others].min() < t[i] < t[others].max()
    n = tx.client_id.nunique()
    rows.append({"split": split, "decoy joins per 100 streams": 100 * joined / n_streams,
                 "in a gap": 100 * inside / n_streams, "at an end": 100 * (joined - inside) / n_streams,
                 "share of Clients with one": clients / n})
    print(rows[-1], flush=True)
print(pd.DataFrame(rows).round(3).to_string(index=False))
