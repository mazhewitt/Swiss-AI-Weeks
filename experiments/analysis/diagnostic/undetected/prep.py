"""Train only: stream detection + labels + per-transaction family evidence, pickled for the other scripts."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from recurring_family.data import load_transactions, load_labels
from recurring_family.streams import detect_stream_payments, _evidence, _words, _FAMILY_WORDS, HOME_MCC, MUSIC_OR_STREAMING
from recurring_family.config import CUTOFF, LABEL_COLUMN
OUT = Path(__file__).parent
tx = load_transactions(Path("data/raw"), "train")
lab = load_labels(Path("data/raw"), "train")
print(type(lab)); print(lab.head() if hasattr(lab, "head") else lab)
tx = tx[tx.timestamp < CUTOFF].reset_index(drop=True)
streams, pays = detect_stream_payments(tx, CUTOFF)
pickle.dump(dict(tx=tx, lab=lab, streams=streams, pays=pays), open(OUT / "prep.pkl", "wb"))
print(len(tx), len(streams), len(pays))
