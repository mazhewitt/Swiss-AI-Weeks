import warnings; warnings.filterwarnings("ignore")
"""Churn-candidate features per stream (from the stream table + stream member payments) and per Client
(stream-derived, plus raw-transaction ones flagged raw_*). Writes stream_feats.pkl and client_feats.pkl."""
import pickle, re
from pathlib import Path
import numpy as np, pandas as pd
from recurring_family.config import CUTOFF
from recurring_family.streams import _period, _evidence, StreamParams

OUT = Path(__file__).parent
D = pickle.load(open(OUT / "prep.pkl", "rb"))
lab, st, tx, members = D["labels"], D["streams"], D["tx"], D["members"]
DAY = pd.Timedelta(days=1)
TOL = StreamParams().amount_tolerance
mem = {(m["client_id"], m["stream_id"]): m for m in members}
refunds = tx[(tx.type == "refund") & (tx.direction == "in")]
ref_by_client = {c: g for c, g in refunds.groupby("client_id")}

def slope(y):
    if len(y) < 3: return np.nan
    x = np.arange(len(y)); return float(np.polyfit(x, y, 1)[0])

rows = []
for r in st.itertuples(index=False):
    m = mem[(r.client_id, r.stream_id)]
    t, a = m["times"], m["amounts"]
    n = len(t); P = r.period_days
    f = dict(client_id=r.client_id, stream_id=r.stream_id)
    dsl = (CUTOFF - t[-1]) / DAY
    f["days_since_last"] = dsl
    f["days_since_first"] = (CUTOFF - t[0]) / DAY
    f["days_to_next"] = (r.next_payment - CUTOFF) / DAY if pd.notna(r.next_payment) else np.nan
    f["overdue"] = dsl / P if n > 1 else np.nan
    if n > 1:
        gaps = np.asarray((t[1:] - t[:-1]) / DAY, float)
        folds = np.maximum(1, np.round(gaps / P))
        f["last_gap_ratio"] = gaps[-1] / P
        f["last_gap_missed"] = float(folds[-1] >= 2)
        f["last_gap_late"] = (gaps[-1] / folds[-1] - P) / P  # >0: last payment late vs period
        f["n_missed"] = float((folds - 1).sum())
        f["missed_rate"] = f["n_missed"] / (folds.sum())
        f["missed_last3"] = float((folds[-3:] - 1).sum())
        folded = gaps / folds
        f["gap_trend"] = slope(folded / P)
        f["gap_recent_minus_all"] = (folded[-3:].mean() - folded.mean()) / P
        f["gap_mad_rel"] = r.gap_mad_days / P
        span_periods = (t[-1] - t[0]) / DAY / P
        f["expected_minus_actual"] = span_periods + 1 - n
        # periods from last payment to cutoff not yet elapsed: phase within period
        f["phase"] = (dsl % P) / P
    la = np.log(np.clip(a, 1e-9, None))
    f["amt_last_ratio"] = a[-1] / np.median(a)
    f["amt_last_logdev"] = abs(la[-1] - np.median(la))
    f["amt_last_change"] = abs(la[-1] - la[-2]) if n > 1 else np.nan
    f["amt_last_up"] = la[-1] - la[-2] if n > 1 else np.nan
    f["amt_trend"] = slope(la)
    f["amt_first_last"] = la[-1] - la[0]
    f["amt_n_changes"] = float((np.abs(np.diff(la)) > 0.01).sum()) if n > 1 else 0.0
    f["amt_last3_changes"] = float((np.abs(np.diff(la[-4:])) > 0.01).sum()) if n > 1 else 0.0
    # refunds: raw refunds of the Client within 7 days after a member payment
    rf = ref_by_client.get(r.client_id)
    full_last = part_last = full_any_last2 = 0.0; last_ref_days = np.nan; n_partial = 0
    ref_near_end_raw = 0.0
    if rf is not None and len(rf):
        rt = pd.DatetimeIndex(rf.timestamp).as_unit("ns").asi8; ti = pd.DatetimeIndex(t).as_unit("ns").asi8; D7 = 7 * 86400 * 10**9; c45 = (CUTOFF - 45 * DAY).value; ra = rf.amount.to_numpy(float)
        ev = [_evidence(d, mc)[0] for d, mc in zip(rf.description, rf.mcc)]
        ok = np.array([e != "drop" for e in ev])
        for k in range(n):
            w = ok & (rt >= ti[k]) & (rt <= ti[k] + D7)
            if not w.any(): continue
            ratio = ra[w] / a[k]
            full = np.abs(np.log(ratio)) <= TOL
            part = (ratio < np.exp(-TOL)) & (ratio > 0.05)
            if full.any() or part.any():
                last_ref_days = (CUTOFF - t[k]) / DAY
            if part.any(): n_partial += 1
            if k == n - 1:
                full_last = float(full.any()); part_last = float(part.any())
            if k >= n - 2 and full.any(): full_any_last2 = 1.0
        # any subscription-like refund in the last 45 days, any amount
        ref_near_end_raw = float((ok & (rt >= c45)).sum())
    f.update(refund_last_full=full_last, refund_last_partial=part_last, refund_last2_full=full_any_last2,
             refund_last_days=last_ref_days, n_partial_refunds=float(n_partial), raw_sub_refunds_45d=ref_near_end_raw)
    f["last_dom"] = t[-1].day
    f["first_dom"] = t[0].day
    f["next_dom"] = r.next_payment.day if pd.notna(r.next_payment) else np.nan
    f["last_desc_specific"] = float(len(set(re.findall(r"[a-z]+", m["descs"][-1].lower()))) and
                                   _evidence(m["descs"][-1], r.mcc)[0] == "family")
    f["last_desc_diff"] = float(m["descs"][-1].lower() != r.description)
    rows.append(f)
sf = pd.DataFrame(rows)
sf = st.merge(sf, on=["client_id", "stream_id"], suffixes=("", "_m"))
sf["ended"] = (~sf.active) & (sf.n_payments >= 3)
sf["end_time"] = sf.last_payment  # for ended streams
pickle.dump(sf, open(OUT / "stream_feats.pkl", "wb"))

# ---------------- client level --------------------------------------------------------
clients = pd.Index(lab.index, name="client_id")
C = pd.DataFrame(index=clients)
C["y"] = lab; C["none"] = (lab == "none").astype(int)
g = sf.groupby("client_id")
C["n_streams"] = g.size().reindex(clients).fillna(0)
C["n_live"] = sf[sf.active].groupby("client_id").size().reindex(clients).fillna(0)
C["n_ended"] = sf[sf.ended].groupby("client_id").size().reindex(clients).fillna(0)
C["n_short"] = sf[sf.n_payments < 3].groupby("client_id").size().reindex(clients).fillna(0)
C["past_churn_rate"] = C.n_ended / (C.n_ended + C.n_live).replace(0, np.nan)
C["n_ended_recent120"] = sf[sf.ended & (sf.last_payment > CUTOFF - 120 * DAY)].groupby("client_id").size().reindex(clients).fillna(0)
C["n_families_live"] = sf[sf.active].groupby("client_id").family.nunique().reindex(clients).fillna(0)
C["n_families_all"] = g.family.nunique().reindex(clients).fillna(0)
C["total_payments"] = g.n_payments.sum().reindex(clients).fillna(0)
C["stream_first_days"] = ((CUTOFF - g.first_payment.min()) / DAY).reindex(clients)
C["stream_last_days"] = ((CUTOFF - g.last_payment.max()) / DAY).reindex(clients)
# stream payments in last 60 days vs earlier average per 60 days (from member payments only)
allt = {}
for m in members: allt.setdefault(m["client_id"], []).extend(m["times"])
def recent_ratio(ts, w=60):
    ts = pd.DatetimeIndex(ts)
    if len(ts) == 0: return np.nan
    start = ts.min(); span = (CUTOFF - start) / DAY
    if span <= w + 30: return np.nan
    rec = (ts >= CUTOFF - w * DAY).sum(); old = (ts < CUTOFF - w * DAY).sum() / ((span - w) / w)
    return rec / old if old > 0 else np.nan
C["stream_activity_ratio60"] = pd.Series({c: recent_ratio(v) for c, v in allt.items()}).reindex(clients)
C["stream_activity_ratio90"] = pd.Series({c: recent_ratio(v, 90) for c, v in allt.items()}).reindex(clients)

# live-stream aggregates: at the soonest-next live stream ("prim_"), plus min / max over live streams
live = sf[sf.active].copy()
FEATS = ["period_days", "median_amount", "amount_cv", "gap_mad_days", "n_payments", "n_refunds", "refund_rate",
         "family_description_share", "days_since_last", "days_since_first", "days_to_next", "overdue",
         "last_gap_ratio", "last_gap_missed", "last_gap_late", "n_missed", "missed_rate", "missed_last3", "gap_trend",
         "gap_recent_minus_all", "gap_mad_rel", "expected_minus_actual", "phase", "amt_last_ratio", "amt_last_logdev",
         "amt_last_change", "amt_last_up", "amt_trend", "amt_first_last", "amt_n_changes", "amt_last3_changes",
         "refund_last_full", "refund_last_partial", "refund_last2_full", "refund_last_days", "n_partial_refunds",
         "raw_sub_refunds_45d", "last_dom", "first_dom", "next_dom", "last_desc_specific", "last_desc_diff"]
prim = live.sort_values(["client_id", "days_to_next", "n_payments"], ascending=[True, True, False]).groupby("client_id").first()
for c in FEATS:
    C["prim_" + c] = prim[c].reindex(clients)
    C["min_" + c] = live.groupby("client_id")[c].min().reindex(clients)
    C["max_" + c] = live.groupby("client_id")[c].max().reindex(clients)
C["prim_family"] = prim.family.reindex(clients).astype(object)
for fam in sorted(st.family.unique()):
    C["live_has_" + fam] = live[live.family == fam].groupby("client_id").size().reindex(clients).fillna(0).clip(upper=1)

# ---------------- raw-transaction client features (decoy-sensitive unless noted) -------------
t = tx.copy()
t["words"] = t.description.astype(str).str.lower()
DECOYS = {"digital order", "merchant charge", "service payment", "card purchase"}
cg = t.groupby("client_id")
C["raw_first_tx_days"] = ((CUTOFF - cg.timestamp.min()) / DAY).reindex(clients)
C["raw_last_tx_days"] = ((CUTOFF - cg.timestamp.max()) / DAY).reindex(clients)
C["raw_n_tx"] = cg.size().reindex(clients)
C["raw_n_decoy"] = t[t.words.isin(DECOYS)].groupby("client_id").size().reindex(clients).fillna(0)
# decoy-safe activity: salary topups, atm, p2p, transfers (never card payments)
safe = t[t.type.isin(["topup", "atm", "p2p_transfer", "transfer"])]
def ratio(df, w):
    out = {}
    for c, gg in df.groupby("client_id"):
        ts = gg.timestamp; span = (CUTOFF - ts.min()) / DAY
        if span <= w + 30: out[c] = np.nan; continue
        rec = (ts >= CUTOFF - w * DAY).sum(); old = (ts < CUTOFF - w * DAY).sum() / ((span - w) / w)
        out[c] = rec / old if old > 0 else np.nan
    return pd.Series(out)
C["raw_safe_activity_ratio60"] = ratio(safe, 60).reindex(clients)
C["raw_safe_activity_ratio30"] = ratio(safe, 30).reindex(clients)
sal = t[t.type == "topup"]
C["raw_last_salary_days"] = ((CUTOFF - sal.groupby("client_id").timestamp.max()) / DAY).reindex(clients)
C["raw_salary_amt"] = sal.groupby("client_id").amount.median().reindex(clients)
shop = t[(t.type == "card_payment") & t.description.str.contains("shop|market|foods|grocery|pharmacy|ride|coffee|dining|hotel", regex=True)]
C["raw_shop_activity_ratio60"] = ratio(shop, 60).reindex(clients)
C["raw_card_activity_ratio60"] = ratio(t[t.type == "card_payment"], 60).reindex(clients)
C["raw_n_refunds"] = t[t.type == "refund"].groupby("client_id").size().reindex(clients).fillna(0)
C["raw_refunds_last45"] = t[(t.type == "refund") & (t.timestamp >= CUTOFF - 45 * DAY)].groupby("client_id").size().reindex(clients).fillna(0)
C["raw_n_fee"] = t[t.type == "fee"].groupby("client_id").size().reindex(clients).fillna(0)
C["raw_currency_n"] = cg.currency.nunique().reindex(clients)
C["raw_main_ccy"] = cg.currency.agg(lambda s: s.value_counts().index[0]).reindex(clients)
# card payments not in any stream, with a subscription-like (non-drop) description, last 60 days (decoy-safe by rule)
in_stream = set(np.concatenate([m["idx"] for m in members]))
cp = t[(t.type == "card_payment")]
ev = np.array([_evidence(d, m)[0] for d, m in zip(cp.description, cp.mcc)])
orphan = cp[(ev != "drop") & ~cp.index.isin(in_stream)]
C["raw_orphan_sub_all"] = orphan.groupby("client_id").size().reindex(clients).fillna(0)
C["raw_orphan_sub_60"] = orphan[orphan.timestamp >= CUTOFF - 60 * DAY].groupby("client_id").size().reindex(clients).fillna(0)
oof = D["oof"]
fam_cols = [c for c in oof.columns if c not in ("fold", "none")]
C["oof_none"] = oof["none"].reindex(clients)
C["oof_1_minus_max"] = 1 - oof[fam_cols].max(axis=1).reindex(clients)
C["oof_max"] = oof[fam_cols].max(axis=1).reindex(clients)
C["fold"] = oof["fold"].reindex(clients)
pickle.dump(C, open(OUT / "client_feats.pkl", "wb"))
print(C.shape, "live clients", (C.n_live > 0).sum())
