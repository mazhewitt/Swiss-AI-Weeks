"""Shared: load prep.pkl, mark stream membership and a raw family per transaction."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from recurring_family.streams import _evidence, _words, _FAMILY_WORDS, HOME_MCC, _SHOP_WORDS, _DECOY_WORDS, _MUSIC_HINTS, _STREAMING_HINTS
OUT = Path(__file__).parent
FAMS = ("cloud", "gym", "insurance", "mobile", "music", "software", "streaming")

def raw_family(desc, mcc):
    """Family named by description keywords (music/streaming by hint), else by home MCC; None if shop/decoy/unknown."""
    w = set(_words(desc))
    if not w or w & _SHOP_WORDS:
        return None
    named = [g for g, k in _FAMILY_WORDS.items() if w & k]
    hint = "music" if w & _MUSIC_HINTS else "streaming" if w & _STREAMING_HINTS else None
    if len(named) == 1:
        g = named[0]
        return hint or "music_or_streaming" if g == "music_or_streaming" else g
    home = HOME_MCC.get(str(mcc))
    if home == "music_or_streaming":
        return hint or "music_or_streaming"
    return home

def load():
    D = pickle.load(open(OUT / "prep.pkl", "rb"))
    tx, lab, st, pays = D["tx"], D["lab"], D["streams"], D["pays"]
    y = lab.set_index("client_id")["target_next_recurring_merchant"].astype(str)
    key = pays.assign(k=1).merge(st[["client_id", "stream_id", "family"]], on=["client_id", "stream_id"])
    key["client_id"] = key.client_id.astype(str)
    tx = tx.copy(); tx["client_id"] = tx.client_id.astype(str)
    tx = tx.merge(key[["client_id", "timestamp", "amount", "family", "stream_id"]].drop_duplicates(["client_id", "timestamp", "amount"]),
                  on=["client_id", "timestamp", "amount"], how="left").rename(columns={"family": "stream_family"})
    tx.loc[~((tx.type == "card_payment") & (tx.direction == "out")), ["stream_family", "stream_id"]] = np.nan
    tx["kind"] = [_evidence(d, m)[0] for d, m in zip(tx.description, tx.mcc)]
    tx["rawfam"] = [raw_family(d, m) for d, m in zip(tx.description, tx.mcc)]
    tx["decoy"] = [bool(set(_words(d)) & _DECOY_WORDS) for d in tx.description]
    st = st.copy(); st["client_id"] = st.client_id.astype(str); st["family"] = st.family.astype(str)
    return tx, y, st
