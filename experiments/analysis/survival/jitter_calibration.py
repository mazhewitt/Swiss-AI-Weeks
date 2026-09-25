"""Ticket 14 step 1: how far does a stream's real next payment land from its projection, and does its own
schedule jitter (`gap_mad_days`) predict that spread?

For every train stream with 3+ payments: drop the last payment, re-summarise the head as the detector does
(`streams._period`, folded MAD), project `head[-1] + period`, and measure the residual of the real last
payment, folded past missed slots (k = round(gap / period) periods). Reads transactions only: it runs under
`data.pseudo_labelling()`, which refuses every label file.

Fits sigma(MAD) = sqrt(a^2 + (b * MAD)^2) to the robust spread (1.4826 * MAD of the residuals) per MAD bin,
weighted by count, and reports the tail shape (normal: std / robust ~ 1.0; Laplace: ~ 1.38).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import kurtosis

from recurring_family import data
from recurring_family.cli import Paths
from recurring_family.ranker import _detected
from recurring_family.streams import StreamParams, _period

OUT = Path("experiments/analysis/survival")
DAY = pd.Timedelta(days=1)
BINS = [-0.01, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]


def robust(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def residuals(paid: pd.DataFrame, params=StreamParams()) -> pd.DataFrame:
    rows = []
    for (client, sid), g in paid.groupby(["client_id", "stream_id"], sort=False):
        t = g["timestamp"].sort_values().to_numpy()
        if len(t) < 3:
            continue
        head = t[:-1]
        gaps = np.asarray((head[1:] - head[:-1]) / DAY, dtype=float)
        period = _period(gaps, params)
        folded = gaps / np.maximum(1, np.round(gaps / period))
        mad = float(np.median(np.abs(folded - period)))
        gap = float((t[-1] - head[-1]) / DAY)
        k = max(1, round(gap / period))
        rows.append(dict(client_id=client, stream_id=sid, n_head=len(head), period=period, mad=mad, k=k, residual=gap - k * period))
    return pd.DataFrame(rows)


def main():
    splits = sys.argv[1:] or ["train"]  # e.g. `train unlabeled`: more history, still no labels
    paths = Paths(Path("."))
    parts = []
    with data.pseudo_labelling():
        for split in splits:
            tx = data.load_transactions(paths.raw, split)
            _, paid = _detected(tx, pd.Index(tx["client_id"].astype(str).unique()))
            parts.append(paid.assign(client_id=split + ":" + paid["client_id"].astype(str)))
    paid = pd.concat(parts, ignore_index=True)
    r = residuals(paid)
    print("splits:", splits)
    r["bin"] = pd.cut(r["mad"], BINS)
    out = {"streams": int(len(r)), "skipped_missed_slot_k>1": int((r["k"] > 1).sum())}
    print(f"{len(r)} streams with 3+ payments; {out['skipped_missed_slot_k>1']} folded past a missed slot")

    def table(sub, name):
        g = sub.groupby("bin", observed=True)["residual"]
        t = pd.DataFrame({"n": g.size(), "robust_sigma": g.apply(robust), "std": g.std(), "median": g.median(),
                          "q10": g.quantile(0.1), "q90": g.quantile(0.9)})
        print(f"\n{name}\n{t.round(2).to_string()}")
        return t

    print("\nAll heads (n_head = 2 has MAD 0 by construction)")
    t_all = table(r, "all")
    informative = r[r["n_head"] >= 3]
    t_inf = table(informative, "heads with 2+ gaps (MAD informative)")
    two_gap = r[r["n_head"] == 2]["residual"].to_numpy()
    out["two_payment_head_robust_sigma"] = robust(two_gap)
    out["two_payment_head_n"] = int(len(two_gap))
    print(f"\nheads of 2 payments (1 gap; the closest proxy for single-payment streams): n={len(two_gap)} "
          f"robust sigma {robust(two_gap):.2f} std {two_gap.std():.2f}")

    # fit sigma(MAD) = sqrt(a^2 + (b MAD)^2) on the informative bins' robust spread at the bin's median MAD
    fit = informative.groupby("bin", observed=True).agg(mad=("mad", "median"), n=("residual", "size"),
                                                       sigma=("residual", lambda x: robust(x.to_numpy())))
    fit = fit[fit["n"] >= 20]

    def resid(p):
        a, b = p
        return np.sqrt(fit["n"]) * (np.sqrt(a**2 + (b * fit["mad"])**2) - fit["sigma"])

    sol = least_squares(resid, x0=[1.0, 1.0], bounds=([0, 0], [np.inf, np.inf]))
    a, b = map(float, sol.x)
    fit["sqrt_form"] = np.sqrt(a**2 + (b * fit["mad"])**2)

    # the alternative: a floor, then proportional: sigma = max(a2, b2 * MAD)
    def resid_max(p):
        a2, b2 = p
        return np.sqrt(fit["n"]) * (np.maximum(a2, b2 * fit["mad"]) - fit["sigma"])

    sol2 = least_squares(resid_max, x0=[3.5, 1.4], bounds=([0, 0], [np.inf, np.inf]))
    a2, b2 = map(float, sol2.x)
    fit["max_form"] = np.maximum(a2, b2 * fit["mad"])
    wrms = lambda col: float(np.sqrt(np.average((fit[col] - fit["sigma"]) ** 2, weights=fit["n"])))
    print(f"\nfit on informative bins:\n{fit.round(2).to_string()}")
    print(f" sqrt form: a = {a:.3f}  b = {b:.3f}  weighted rms {wrms('sqrt_form'):.3f}")
    print(f" max form:  a = {a2:.3f}  b = {b2:.3f}  weighted rms {wrms('max_form'):.3f}")
    out.update(sqrt_form=dict(a=round(a, 3), b=round(b, 3), wrms=round(wrms("sqrt_form"), 3)),
               max_form=dict(a=round(a2, 3), b=round(b2, 3), wrms=round(wrms("max_form"), 3)),
               fit_bins=fit.reset_index().astype({"bin": str}).round(3).to_dict("records"))

    # does the spread depend on the period (biweekly / monthly / longer)?
    r["period_bin"] = pd.cut(r["period"], [0, 20, 45, 120, np.inf], labels=["biweekly", "monthly", "quarterly", "longer"])
    g = r.groupby("period_bin", observed=True)["residual"]
    by_period = pd.DataFrame({"n": g.size(), "robust_sigma": g.apply(lambda x: robust(x.to_numpy())), "std": g.std()})
    print(f"\nby period:\n{by_period.round(2).to_string()}")
    out["by_period"] = by_period.reset_index().astype({"period_bin": str}).round(3).to_dict("records")

    # tail shape, on residuals standardised by their fitted sigma
    z = informative["residual"] / np.sqrt(a**2 + (b * informative["mad"])**2)
    z = z[z.abs() < 20]
    shape = {"std_over_robust": round(float(z.std() / robust(z.to_numpy())), 3),
             "excess_kurtosis": round(float(kurtosis(z)), 2),
             "within_1sigma": round(float((z.abs() <= 1).mean()), 3), "within_2sigma": round(float((z.abs() <= 2).mean()), 3),
             "normal_reference": {"std_over_robust": 1.0, "within_1sigma": 0.683, "within_2sigma": 0.954},
             "laplace_reference": {"std_over_robust": 1.376, "within_1sigma": 0.757, "within_2sigma": 0.940}}
    print(f"\ntail shape of standardised residuals: {json.dumps(shape)}")
    out["tail"] = shape
    out["splits"] = splits
    name = "jitter_calibration" + ("" if splits == ["train"] else "_" + "+".join(splits)) + ".json"
    (OUT / name).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
