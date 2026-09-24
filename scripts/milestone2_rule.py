"""Milestone-2 submission: E1 soonest-next-payment rule plus a `none`-gate.

For each Client, take Active Streams with 3+ payments whose projected next
payment falls inside the Horizon, and predict the family of the soonest one.
If the Client's longest such stream has `max_last_n` or fewer payments, or no
stream qualifies, predict `none`.

Train macro-F1 with max_last_n=4: 0.5331.

Usage (from the repo root):
    PYTHONPATH=src uv run python scripts/milestone2_rule.py score
    PYTHONPATH=src uv run python scripts/milestone2_rule.py write submissions/milestone2_rules_none_gate_v2.csv
"""

import sys
from pathlib import Path

from sklearn.metrics import f1_score

from recurring_family import data
from recurring_family.config import CUTOFF, LABELS
from recurring_family.streams import cached_streams

RAW = Path("data/raw")
CACHE = Path("artifacts/streams")
HORIZON_DAYS = 90
MAX_LAST_N = 4


def predict(split, clients, min_n=3, max_last_n=MAX_LAST_N):
    streams, _ = cached_streams(RAW, split, CACHE)
    live = streams[streams.active & (streams.n_payments >= min_n)].copy()
    live["days_to_next"] = (live.next_payment - CUTOFF).dt.total_seconds() / 86400
    live = live[live.days_to_next <= HORIZON_DAYS]
    if max_last_n is not None:
        # none-gate: Clients whose longest live stream is short are mostly `none`
        longest = live.groupby("client_id").n_payments.transform("max")
        live = live[longest > max_last_n]
    soonest = live.sort_values("days_to_next").groupby("client_id").family.first()
    return soonest.reindex(clients).fillna("none").astype(str)


def score(split, **kw):
    labels = data.load_labels(RAW, split).set_index("client_id").iloc[:, -1].astype(str)
    return f1_score(labels, predict(split, labels.index, **kw), labels=list(LABELS), average="macro")


def write(out_path):
    clients = data.load_sample_submission(RAW).client_id
    out = predict("test", clients).rename("predicted_next_recurring_merchant")
    out.index.name = "client_id"
    out.reset_index().to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows to {out_path}")
    print(out.value_counts().to_string())


if __name__ == "__main__":
    if sys.argv[1:2] == ["write"]:
        write(sys.argv[2])
    else:
        for kw in [dict(max_last_n=None), dict(max_last_n=3), dict(max_last_n=4), dict(max_last_n=5)]:
            print("train", kw, round(score("train", **kw), 4))
