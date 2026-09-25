"""Ticket 23, step 1 (label-free): description and MCC noise of the payments on a stream's schedule, per split,
and the corruption parameters derived from test against train.

Reads transactions only, through the project's loader (`data.load_transactions`), for train, valid (all 1,000
Clients) and test; never a label file. Streams come from the project's detector (`schedule.py` says exactly
which payments count as on a stream's schedule).

Writes `noise_params.json` (the corruption parameters, fixed before step 2) and the `step1` section of
`results.json`.

    uv run python experiments/analysis/noise_regime/measure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recurring_family import data  # noqa: E402
from recurring_family.config import MERCHANT_FAMILIES  # noqa: E402
from schedule import KINDS, home_mcc, schedule_payments, summarise  # noqa: E402

RAW = Path("data/raw")
OUT = Path("experiments/analysis/noise_regime")
SPLITS = ("train", "valid", "test")


def update_results(key: str, value) -> None:
    path = OUT / "results.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    result[key] = value
    path.write_text(json.dumps(result, indent=1))


def noise_draws(payments: pd.DataFrame, streams: pd.DataFrame) -> dict:
    """Per family, the empirical distributions noisy descriptions and MCCs are drawn from: the raw
    description strings of on-schedule payments whose kind is not `own`, and the MCCs that are not the
    family's home MCC. A `slot` payment is weighted by the share of its kind's slot hits that are not
    chance (the control), so chance hits do not shape the draws."""
    ratio = streams["on_slots"].sum() / streams["control_slots"].sum()
    on = payments[payments["role"] != "control"].copy()
    ctrl = payments[payments["role"] == "control"]
    slot_n = on[on["role"] == "slot"].groupby("kind").size().reindex(KINDS, fill_value=0)
    chance = ctrl.groupby("kind").size().reindex(KINDS, fill_value=0) * ratio
    keep = (1 - chance / slot_n.where(slot_n > 0)).clip(lower=0).fillna(0)
    on["w"] = 1.0
    slot = on["role"] == "slot"
    on.loc[slot, "w"] = on.loc[slot, "kind"].map(keep).astype(float)
    out = {}
    for f in MERCHANT_FAMILIES:
        g = on[on["family"] == f]
        d = g[g["kind"] != "own"].groupby("description")["w"].sum()
        m = g[g["mcc_noise"]].groupby("mcc")["w"].sum()
        d, m = d[d > 0] / d.sum(), m[m > 0] / m.sum()
        out[f] = {
            "description": {k: round(float(v), 6) for k, v in d.sort_values(ascending=False).items()},
            "mcc": {k: round(float(v), 6) for k, v in m.sort_values(ascending=False).items()},
            "description_kind": {
                k: round(float(g.loc[g["kind"] == k, "w"].sum() / g.loc[g["kind"] != "own", "w"].sum()), 4)
                for k in KINDS[1:]
            },
        }
    return out


def corruption_rate(noisy: float, clean: float) -> float:
    """Replacing a payment's value by a noise draw with probability c (a draw is always noisy) moves the
    noisy share from `clean` to clean + c (1 - clean); c is the rate that reaches `noisy`."""
    return max(0.0, (noisy - clean) / (1 - clean))


def main() -> None:
    summary, draws = {}, {}
    for split in SPLITS:
        tx = data.load_transactions(RAW, split)
        payments, streams, table = schedule_payments(tx)
        summary[split] = summarise(payments, streams, table, tx["client_id"].nunique())
        if split == "test":
            draws = noise_draws(payments, streams)
        s = summary[split]["all"]
        print(f"{split}: description noise {s['description_noise']:.4f} {s['description']}  "
              f"MCC noise {s['mcc_noise']:.4f}  {summary[split]['fragmentation']}", flush=True)
    rates = {}
    for f in MERCHANT_FAMILIES:
        tr, te = summary["train"]["by_family"][f], summary["test"]["by_family"][f]
        va = summary["valid"]["by_family"][f]
        rates[f] = {
            "description": round(corruption_rate(te["description_noise"], tr["description_noise"]), 4),
            "mcc": round(corruption_rate(te["mcc_noise"], tr["mcc_noise"]), 4),
            "valid_description": round(corruption_rate(va["description_noise"], tr["description_noise"]), 4),
            "valid_mcc": round(corruption_rate(va["mcc_noise"], tr["mcc_noise"]), 4),
            "home_mcc": home_mcc(f),
        }
    params = {
        "about": (
            "Ticket 23 corruption parameters, derived from test against train (label-free) and fixed before step 2. "
            "A train payment that belongs to a detected stream of two or more payments (whole history, default "
            "detector) of family F gets, independently: its description replaced with probability "
            "rates[F].description by a draw from draws[F].description, and its MCC replaced with probability "
            "rates[F].mcc by a draw from draws[F].mcc. Dose d multiplies both rates. valid_* rates are for "
            "reference only."
        ),
        "rate_formula": "c = (test_share - train_share) / (1 - train_share), per family, on-schedule payments",
        "rates": rates,
        "draws": draws,
    }
    (OUT / "noise_params.json").write_text(json.dumps(params, indent=1))
    update_results("step1", summary)
    print(json.dumps(rates, indent=1))


if __name__ == "__main__":
    main()
