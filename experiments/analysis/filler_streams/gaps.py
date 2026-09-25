"""Label-free (ticket 11): which payments left out of every stream sit at a detected stream's amount and on its
schedule (in a gap where a payment is missing, or one or two periods beyond its ends), by `_evidence` kind,
per split; plus an off-phase control (half a period off the schedule) and streams split in two.
Reads transactions only.

    uv run python experiments/analysis/filler_streams/gaps.py [TOLERANCE_DAYS=4] [SPLITS=train,valid,test]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from recurring_family import data
from recurring_family.config import CUTOFF
from recurring_family.streams import StreamParams, _detect_per_client

from kindlib import kind_of

P = StreamParams()
TOL_DAYS = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
SPLITS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["train", "valid", "test"]
pd.set_option("display.width", 250)


rows_gap, rows_stream, rows_split = [], [], []
for split in SPLITS:
    tx = data.load_transactions(Path("data/raw"), split)
    n_clients = tx.client_id.nunique()
    for client, pay, srows, membership, _ in _detect_per_client(tx, CUTOFF, P):
        t = pd.DatetimeIndex(pay["timestamp"]).as_unit("ns").asi8 / 86400e9
        la = np.log(pay["amount"].to_numpy(float).clip(1e-9))
        kinds = None
        free = membership < 0
        by_fam = {}
        for r in srows:
            s = r["stream_id"]
            m = np.flatnonzero(membership == s)
            m = m[np.argsort(t[m], kind="stable")]
            by_fam.setdefault(r["family"], []).append((r, t[m]))
            per = r["period_days"]
            n = len(m)
            if n < 2 or not np.isfinite(per):
                continue
            gaps = np.diff(t[m])
            folds = np.maximum(1, np.round(gaps / per))
            n_missed = int((folds - 1).sum())
            lo, hi = la[m].min() - P.amount_tolerance, la[m].max() + P.amount_tolerance
            cand = np.flatnonzero(free & (la >= lo) & (la <= hi))
            found = {}
            for j in cand:
                tj = t[j]
                # expected dates: every member + k*period, k in 1..(fold-1) inside gaps; ends +/- 1..2 periods
                where = None
                for a, b, f in zip(t[m][:-1], t[m][1:], folds):
                    if f >= 2 and a < tj < b:
                        exp = a + per * np.arange(1, f)
                        if np.min(np.abs(exp - tj)) <= TOL_DAYS:
                            where = "gap"
                        elif np.min(np.abs(exp - per / 2 - tj)) <= TOL_DAYS:
                            where = "gap_control_offphase"
                        break
                if where is None and tj > t[m][-1]:
                    k = np.round((tj - t[m][-1]) / per)
                    if 1 <= k <= 2 and abs(tj - t[m][-1] - k * per) <= TOL_DAYS:
                        where = "after_last"
                if where is None and tj < t[m][0]:
                    k = np.round((t[m][0] - tj) / per)
                    if 1 <= k <= 2 and abs(t[m][0] - tj - k * per) <= TOL_DAYS:
                        where = "before_first"
                if where:
                    if kinds is None:
                        kinds = [kind_of(d, mm) for d, mm in zip(pay["description"], pay["mcc"])]
                    found[j] = where
                    rows_gap.append({"split": split, "client_id": client, "stream_id": s, "family": r["family"],
                                     "where": where, "kind": kinds[j], "mcc": pay["mcc"].iloc[j],
                                     "description": pay["description"].iloc[j], "active": r["active"]})
            rows_stream.append({"split": split, "client_id": client, "stream_id": s, "n": n, "active": r["active"],
                                "n_missed": n_missed, "missed_rate": n_missed / folds.sum(),
                                "n_gap_fill": sum(v == "gap" for v in found.values())})
        # split streams: same family, amounts within tolerance, one ends before the other starts on schedule
        for fam, ss in by_fam.items():
            for i, (r1, t1) in enumerate(ss):
                for r2, t2 in ss:
                    if r2 is r1 or t2[0] <= t1[-1]:
                        continue
                    if abs(np.log(r1["median_amount"]) - np.log(r2["median_amount"])) > 3 * P.amount_tolerance:
                        continue
                    per = r1["period_days"] if np.isfinite(r1["period_days"]) else r2["period_days"]
                    if not np.isfinite(per):
                        continue
                    k = np.round((t2[0] - t1[-1]) / per)
                    if 1 <= k <= 3 and abs(t2[0] - t1[-1] - k * per) <= TOL_DAYS:
                        rows_split.append({"split": split, "client_id": client, "family": fam,
                                           "amt1": r1["median_amount"], "amt2": r2["median_amount"],
                                           "n1": r1["n_payments"], "n2": r2["n_payments"], "k": k,
                                           "same_amount": abs(np.log(r1["median_amount"]) - np.log(r2["median_amount"])) <= P.amount_tolerance})
    print(split, "done", flush=True)

gap = pd.DataFrame(rows_gap)
st = pd.DataFrame(rows_stream)
sp = pd.DataFrame(rows_split)
n_streams = st.groupby("split").size()
print("\nstreams with >=2 payments:", n_streams.to_dict())
print("missed payments per stream (all / active):")
print(st.groupby("split").agg(missed=("n_missed", "mean"), missed_rate=("missed_rate", "mean"), gap_fill=("n_gap_fill", "mean")).round(4))
print("\nrecoverable payments per 100 streams, by kind and position:")
tab = gap.groupby(["split", "kind", "where"]).size().unstack("split").fillna(0)
print((tab / n_streams * 100).round(2).to_string())
print("\nby kind (all positions), per 100 streams:")
print((gap.groupby(["kind", "split"]).size().unstack("split").fillna(0) / n_streams * 100).round(2).to_string())
if len(sp):
    print("\nsplit streams (same family, on schedule) per 100 streams:")
    print((sp.groupby(["split", "same_amount"]).size().unstack("split").fillna(0) / n_streams * 100).round(2).to_string())
for split in SPLITS:
    g = gap[(gap.split == split) & (gap.kind == "drop:filler-offhome")]
    print(split, "drop:filler-offhome MCCs", g.mcc.value_counts().head(6).to_dict())
