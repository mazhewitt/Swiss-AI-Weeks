"""Load train (under the training_run guard), detect streams at the Cutoff with payment membership,
and save: labels, OOF, stream table, per-stream member payments, per-client raw tx summary.
Train only; valid labels are never read."""
import pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from recurring_family import data
from recurring_family.config import CUTOFF, LABEL_COLUMN
from recurring_family.streams import _detect_per_client, StreamParams, cached_streams, COLUMNS, detect_streams

OUT = Path(__file__).parent
RAW = Path("data/raw")
with data.training_run():
    labels = data.load_labels(RAW, "train").set_index("client_id")[LABEL_COLUMN].astype(str)
    tx = data.load_transactions(RAW, "train")
t0 = time.time()
streams, members = [], []
for client, pays, rows, membership, _ in _detect_per_client(tx, CUTOFF, StreamParams()):
    ts = pays["timestamp"].to_numpy(); am = pays["amount"].to_numpy(float); ds = pays["description"].astype(str).to_numpy()
    for r in rows:
        streams.append(r)
        m = np.flatnonzero(membership == r["stream_id"])
        o = np.lexsort((am[m], ts[m])); m = m[o]
        members.append(dict(client_id=client, stream_id=r["stream_id"], times=pd.DatetimeIndex(ts[m]), amounts=am[m], descs=ds[m],
                            idx=pays.index.to_numpy()[m]))
print("detect", time.time() - t0)
st = pd.DataFrame(streams)[COLUMNS]
cached, _ = cached_streams(RAW, "train", Path("artifacts/streams"))
cmp = cached.sort_values(["client_id", "stream_id"]).reset_index(drop=True)
chk = st.sort_values(["client_id", "stream_id"]).reset_index(drop=True)
assert len(cmp) == len(chk) and (cmp.n_payments.to_numpy() == chk.n_payments.to_numpy()).all(), "mismatch vs cache"
oof = pd.read_csv(OUT / "oof_ranker.csv", dtype={"client_id": str}).set_index("client_id")
pickle.dump(dict(labels=labels, tx=tx, streams=cached.sort_values(["client_id","stream_id"]).reset_index(drop=True),
                 members=members, oof=oof), open(OUT / "prep.pkl", "wb"))
print("saved", len(cached), "streams", len(members))
