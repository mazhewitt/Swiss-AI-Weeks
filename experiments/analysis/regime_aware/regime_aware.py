"""Ticket 24: a regime-aware file, after the competition. Hail mary configuration A (v2 plus the hard race,
E3 on the weight-1 out-of-fold probabilities) with the valid Clients at weight 3 (ticket 21's method), in
four candidates fixed before any number:

- `B`     ticket 21 exactly; its frozen rehearsal predictions are `target_weight/predictions/rehearsal_w3.csv`.
- `R1`    train in the test regime: train's and the unlabeled split's transactions corrupted to test's
          measured noise (ticket 23's `noise_params.json`, dose 1x, seed 0, the same per-payment draws); valid
          is left as it is; the Pseudo-Labels are rebuilt from the corrupted data.
- `R2`    the noise-robust stream detector (`StreamParams(robust=True)`) everywhere: the fit, the target,
          the `none` model and the Pseudo-Labels.
- `R1R2`  both.

**Corruption.** As ticket 23's `regime.py stage_corrupt` at dose 1: the payments of every stream of two or
more payments that the default detector finds in the split's history before the Cutoff are the stream
payments; each gets four uniform draws from one seed-0 generator, in transaction order (u_desc, u_desc_pick,
u_mcc, u_mcc_pick), and its description (MCC) is replaced when u_desc < rate[F].description (u_mcc <
rate[F].mcc) by the inverse-CDF draw from draws[F]. Train and unlabeled each get their own seed-0 generator.
Train's corrupted table equals ticket 23's dose-1 table (its content hash is checked).

**Caches.** The stream and Pseudo-Label caches of `rf` are keyed on the raw file's size and mtime, so they
would serve clean tables: every Pseudo-Label and stream table here is detected from the DataFrame
(`pseudo` stage), as `regime.py` does. For `B`'s settings the bypass reproduces the rows `rf` pools.

Stages (cached under artifacts/regime_aware/<candidate>/<domain>/, so they run as parallel processes):

    corrupt                           train and unlabeled at dose 1 (seed 0)
    pseudo <candidate|B>              the candidate's Pseudo-Labelled rows (B: the bypass check only)
    oof <candidate> <h0|h1|unsealed> <v2|surv>
                                      weight-1 5-fold out-of-fold probabilities on the domain's labelled Clients
    fit <candidate> <h0|h1|unsealed> <v2|surv>
                                      fit at weight 3, predict the domain's target (a half, or test)
    predict <candidate>               freeze the candidate's 700 rehearsal predictions (no label read to score)
    score                             the only stage that reads the selection labels to score (data.scoring())
    final                             the chosen candidate on the unsealed domain: the submission and checks

    uv run python experiments/analysis/regime_aware/regime_aware.py <stage> ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "target_weight"))
sys.path.insert(0, str(HERE.parent / "hailmary"))
import hailmary as hm  # noqa: E402
import target_weight as tw  # noqa: E402

from recurring_family import data  # noqa: E402
from recurring_family import decision as decision_layer  # noqa: E402
from recurring_family.cli import _labeller  # noqa: E402
from recurring_family.config import CUTOFF, LABEL_COLUMN, LABELS, NONE_LABEL, SHIFTED_CUTOFF  # noqa: E402
from recurring_family.cross_validation import out_of_fold_proba  # noqa: E402
from recurring_family.evaluation import paired_bootstrap, score  # noqa: E402
from recurring_family.none_model import NoneModel  # noqa: E402
from recurring_family.ranker import NONE_FOLDS, RANKER_NONE, RankerModel, pseudo_examples  # noqa: E402
from recurring_family.streams import StreamParams, _detect_per_client, detect_streams, pseudo_label_sweep  # noqa: E402
from recurring_family.submission import write_submission  # noqa: E402
from recurring_family.survival import SurvivalModel  # noqa: E402

PATHS = hm.PATHS
RAW = PATHS.raw
OUT = Path("experiments") / "analysis" / "regime_aware"
CACHE = PATHS.artifacts / "regime_aware"
NOISE = Path("experiments") / "analysis" / "noise_regime"
SEED = 0
PSEUDO_MIN_PAYMENTS = 4
PSEUDO_WEIGHT = 0.5
TRAIN_DOSE1_HASH = "a7708bc9684a"  # ticket 23's dose-1 train (noise_regime/results.json)
CANDIDATES = {"R1": (True, False), "R2": (False, True), "R1R2": (True, True)}  # (corrupt, robust)
ALL = ("B", *CANDIDATES)
B_REHEARSAL = Path("experiments") / "analysis" / "target_weight" / "predictions" / "rehearsal_w3.csv"
SUBMISSION = "day2_postmortem_regime"
UNSEALED_FILE = PATHS.submissions / "day2_final_unsealed.csv"
DOMAINS = ("h0", "h1", "unsealed")


def cache(candidate: str, domain: str | None = None) -> Path:
    path = CACHE / candidate / domain if domain else CACHE / candidate
    path.mkdir(parents=True, exist_ok=True)
    return path


def content_hash(tx: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(tx[sorted(tx.columns)], index=False).to_numpy().tobytes()).hexdigest()[:12]


def stream_params(candidate: str) -> StreamParams | None:
    """The robust detector's settings for R2 and R1R2; None (every model's default) otherwise."""
    return StreamParams(robust=True) if CANDIDATES.get(candidate, (False, False))[1] else None


# --- corruption (R1) ----------------------------------------------------------------------------------


def _pick(u: float, dist: dict) -> str:
    keys = list(dist)
    cdf = np.cumsum([dist[k] for k in keys])
    return keys[min(int(np.searchsorted(cdf / cdf[-1], u, side="right")), len(keys) - 1)]


def corrupt(clean: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict]:
    """`clean` at dose 1, exactly as ticket 23's `regime.py stage_corrupt` (one seed-0 generator)."""
    rows, fams = [], []
    for _, pay, srows, membership, _ in _detect_per_client(clean, CUTOFF, StreamParams()):
        family = {r["stream_id"]: r["family"] for r in srows if r["n_payments"] >= 2}
        for idx, m in zip(pay.index, membership):
            if m in family:
                rows.append(idx)
                fams.append(family[m])
    order = np.argsort(rows, kind="stable")
    rows, fams = np.asarray(rows)[order], np.asarray(fams, dtype=object)[order]
    u = np.random.default_rng(SEED).random((len(rows), 4))  # u_desc, u_desc_pick, u_mcc, u_mcc_pick
    desc_draw = np.array([_pick(p, params["draws"][f]["description"]) for p, f in zip(u[:, 1], fams)], dtype=object)
    mcc_draw = np.array([_pick(p, params["draws"][f]["mcc"]) for p, f in zip(u[:, 3], fams)], dtype=object)
    rate_d = np.array([params["rates"][f]["description"] for f in fams])
    rate_m = np.array([params["rates"][f]["mcc"] for f in fams])
    tx = clean.copy()
    hit_d, hit_m = u[:, 0] < rate_d, u[:, 2] < rate_m
    tx.loc[rows[hit_d], "description"] = desc_draw[hit_d]
    tx.loc[rows[hit_m], "mcc"] = mcc_draw[hit_m]
    tx = tx.astype(data.TRANSACTION_DTYPES)
    info = {
        "transactions": int(len(clean)), "stream_payments": int(len(rows)),
        "description_replaced": int(hit_d.sum()), "mcc_replaced": int(hit_m.sum()),
        "description_changed": int((tx["description"] != clean["description"]).sum()),
        "mcc_changed": int((tx["mcc"] != clean["mcc"]).sum()), "content_hash": content_hash(tx),
    }
    return tx, info


def corrupted_path(split: str) -> Path:
    return cache("corrupted") / f"{split}_dose1.pkl"


def stage_corrupt() -> None:
    params = json.loads((NOISE / "noise_params.json").read_text())
    info = {}
    for split in ("train", "unlabeled"):
        tx, info[split] = corrupt(data.load_transactions(RAW, split), params)
        tx.to_pickle(corrupted_path(split))
        print(split, info[split], flush=True)
    assert info["train"]["content_hash"] == TRAIN_DOSE1_HASH, "train does not reproduce ticket 23's dose-1 table"
    (cache("corrupted") / "info.json").write_text(json.dumps(info, indent=1))


def split_transactions(split: str, corrupted: bool) -> pd.DataFrame:
    return pd.read_pickle(corrupted_path(split)) if corrupted else data.load_transactions(RAW, split)


# --- Pseudo-Labels, from the DataFrame (never the size-and-mtime caches) -------------------------------


def build_pseudo(corrupted: bool, params: StreamParams) -> tuple[pd.DataFrame, dict]:
    """v2's Pseudo-Labelled rows: train then unlabeled at the Shifted Cutoff, min_payments 4, as `rf` pools
    them, from the (corrupted) DataFrames and the given detector."""
    labeller = _labeller(PSEUDO_MIN_PAYMENTS, None, None)
    parts, info = [], {}
    for split in ("train", "unlabeled"):
        tx = split_transactions(split, corrupted)
        with data.pseudo_labelling():
            labels = pseudo_label_sweep(tx, (labeller,), SHIFTED_CUTOFF, params=params)[labeller]
            streams = detect_streams(tx, cutoff=SHIFTED_CUTOFF, params=params)
        rows = pseudo_examples(streams, labels, SHIFTED_CUTOFF)
        rows["client_id"] = f"{split}@{SHIFTED_CUTOFF:%Y-%m-%d}:" + rows["client_id"].astype(str)
        parts.append(rows)
        info[split] = {
            "clients": int(len(labels)), "rows": int(len(rows)), "none_share": round(float((labels == NONE_LABEL).mean()), 4),
            "label_mix": {k: int(v) for k, v in labels.value_counts().items()},
        }
    return pd.concat(parts, ignore_index=True), info


def stage_pseudo(candidate: str) -> None:
    if candidate == "B":  # the bypass reproduces the rows `rf` pools from its caches, exactly
        mine, _ = build_pseudo(False, StreamParams())
        hm.v2_factory()
        reference, _ = hm._PSEUDO
        pd.testing.assert_frame_equal(mine.reset_index(drop=True), reference.reset_index(drop=True))
        print("pseudo B: the DataFrame bypass equals rf's pooled rows", flush=True)
        return
    corrupted, _ = CANDIDATES[candidate]
    rows, info = build_pseudo(corrupted, stream_params(candidate) or StreamParams())
    rows.to_pickle(cache(candidate) / "pseudo.pkl")
    (cache(candidate) / "pseudo.json").write_text(json.dumps(info, indent=1))
    print(f"pseudo {candidate}: {len(rows)} rows {info}", flush=True)


# --- models -------------------------------------------------------------------------------------------


class GroupedRanker(tw.GroupedRanker):
    """Ticket 21's `GroupedRanker` (a Client's duplicates share its `none` cross-fit fold), with the inner
    rankers on the same stream detector as the outer one. Without `stream_params` it is ticket 21's."""

    def _cross_fitted_none(self, transactions: pd.DataFrame, labels: pd.Series) -> pd.Series:
        if getattr(self, "stream_params", None) is None:
            return super()._cross_fitted_none(transactions, labels)
        clients = pd.Index(labels.index.astype(str), name="client_id")
        out = pd.Series(np.nan, index=clients, name=RANKER_NONE)
        if len(clients) < 2:
            return out
        bases = pd.Index([tw._base(c) for c in clients])
        unique = pd.Index(pd.unique(bases))
        splitter = KFold(n_splits=min(NONE_FOLDS, len(unique)), shuffle=True, random_state=0)
        for fit_b, score_b in splitter.split(unique):
            fit_idx = np.flatnonzero(bases.isin(set(unique[fit_b])))
            score_idx = np.flatnonzero(bases.isin(set(unique[score_b])))
            ranker = RankerModel(pseudo=self.pseudo, pseudo_weight=self.pseudo_weight, stream_params=self.stream_params)
            ranker.fit(transactions, labels.iloc[fit_idx])
            out.iloc[score_idx] = ranker.predict_proba(transactions, labels.index[score_idx])[NONE_LABEL].to_numpy()
        return out


def v2_factory(candidate: str):
    pseudo = pd.read_pickle(cache(candidate) / "pseudo.pkl")
    kw = {} if stream_params(candidate) is None else {"stream_params": stream_params(candidate)}
    return lambda: GroupedRanker(pseudo=pseudo, pseudo_weight=PSEUDO_WEIGHT, none_model=NoneModel(), **kw)


def surv_factory(candidate: str):
    kw = {} if stream_params(candidate) is None else {"stream_params": stream_params(candidate)}
    return lambda: SurvivalModel(order="unprojected-last", **kw)


FACTORIES = {"v2": v2_factory, "surv": surv_factory}


# --- data ---------------------------------------------------------------------------------------------


def fit_set(candidate: str, domain: str, weight: int) -> tuple[pd.DataFrame, pd.Series]:
    """Ticket 21's (22's for `unsealed`) fit set, with train's rows corrupted for R1 and R1R2. Train's rows
    come first in it, in the raw file's order, so they are swapped for the corrupted table's."""
    tx, labels = tw.fit_set(domain, weight)
    if CANDIDATES[candidate][0]:
        bad = split_transactions("train", True)
        n = len(bad)
        assert (tx["client_id"].iloc[:n].to_numpy() == bad["client_id"].to_numpy()).all()
        assert not set(tx["client_id"].iloc[n:].astype(str)) & set(bad["client_id"].astype(str))
        tx = pd.concat([bad, tx.iloc[n:]], ignore_index=True)
    return tx, labels


def training_guard(domain: str):
    return tw.training_guard(domain)


def target(domain: str) -> tuple[pd.DataFrame, pd.Index]:
    return tw.target("unsealed" if domain == "unsealed" else domain)


# --- stages -------------------------------------------------------------------------------------------


def stage_oof(candidate: str, domain: str, model: str) -> None:
    make = FACTORIES[model](candidate)
    tx, labels = fit_set(candidate, domain, 1)
    with training_guard(domain):
        oof = out_of_fold_proba(make, tx, labels)
    oof.to_csv(cache(candidate, domain) / f"oof_{model}.csv")
    print(f"oof {candidate} {domain} {model}: {len(oof)} Clients", flush=True)


def stage_fit(candidate: str, domain: str, model: str) -> None:
    make = FACTORIES[model](candidate)
    tx, labels = fit_set(candidate, domain, 3)
    with training_guard(domain):
        fitted = make().fit(tx, labels)
    target_tx, clients = target(domain)
    with data.predicting():
        proba = fitted.predict_proba(target_tx, clients)[list(LABELS)]
    proba.to_csv(cache(candidate, domain) / f"target_{model}_w3.csv", index_label="client_id")
    print(f"fit {candidate} {domain} w=3 {model}: predicted {len(proba)} target Clients", flush=True)


def averaged(candidate: str, domain: str) -> pd.DataFrame:
    v2 = hm.read_proba(cache(candidate, domain) / "target_v2_w3.csv")
    surv = hm.read_proba(cache(candidate, domain) / "target_surv_w3.csv")
    return (v2 + surv.reindex(v2.index)) / 2


def decision(candidate: str, domain: str) -> decision_layer.Decision:
    """E3 on the weight-1 out-of-fold probabilities of the candidate's fit set (its labels are training ones)."""
    c = cache(candidate, domain)
    oof = (hm.read_proba(c / "oof_v2.csv") + hm.read_proba(c / "oof_surv.csv")) / 2
    _, labels = fit_set(candidate, domain, 1)
    labels = pd.Series(labels.astype(str).to_numpy(), index=labels.index.astype(str))
    fitted, _ = decision_layer.fit(oof, labels.reindex(oof.index))
    return fitted


def stage_predict(candidate: str) -> None:
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    pred = pd.concat([decision(candidate, d).apply(averaged(candidate, d)) for d in ("h0", "h1")]).sort_index()
    assert len(pred) == 700 and pred.notna().all() and pred.index.is_unique
    pred.rename("predicted").to_csv(OUT / "predictions" / f"rehearsal_{candidate}.csv", index_label="client_id")
    print(f"predict {candidate}: frozen; label mix {pred.value_counts().to_dict()}", flush=True)


def _read(path: Path) -> pd.Series:
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("client_id")["predicted"]


def rehearsal_predictions() -> dict[str, pd.Series]:
    return {"B": _read(B_REHEARSAL), **{c: _read(OUT / "predictions" / f"rehearsal_{c}.csv") for c in CANDIDATES}}


def stage_score() -> None:
    preds = rehearsal_predictions()  # every candidate's predictions are frozen before a label is read
    with data.scoring():
        truth = data.load_labels(RAW, "selection").set_index("client_id")[LABEL_COLUMN]
    truth.index = truth.index.astype(str)
    truth = truth.astype(str)
    assert all(set(p.index) == set(truth.index) for p in preds.values())
    rows = {}
    for name, p in preds.items():
        p = p.reindex(truth.index)
        s = score(truth, p)
        row = {"macro_f1": round(s["macro_f1"], 4), **{f"f1_{l}": round(s[f"f1_{l}"], 4) for l in LABELS}}
        if name != "B":
            cmp = paired_bootstrap(truth, p, preds["B"].reindex(truth.index))
            row["vs_B"] = {k: round(v, 4) for k, v in cmp.items()}
            row["agree_B"] = round(float((p == preds["B"].reindex(truth.index)).mean()), 4)
        row["none_predicted"] = int((p == NONE_LABEL).sum())
        rows[name] = row
    raw = {n: score(truth, p.reindex(truth.index))["macro_f1"] for n, p in preds.items()}
    best = max(ALL, key=lambda n: (raw[n], n == "B"))  # a tie keeps B
    chosen = best if raw[best] > raw["B"] else "B"
    result = {
        "rehearsal": rows,
        "chosen": chosen,
        "chosen_note": (
            "best of 4 on the same 700 selection labels (pooled macro-F1), so its rehearsal score is optimistic"
            if chosen != "B" else "no candidate beat B on the pooled rehearsal: B"
        ),
        "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = OUT / "results.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(result)
    path.write_text(json.dumps(old, indent=1))
    print(json.dumps(result, indent=1))


def _file_labels(path: Path, expected: pd.Series) -> pd.Series:
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("client_id").iloc[:, 0].reindex(expected)


def stage_final() -> None:
    path = OUT / "results.json"
    result = json.loads(path.read_text())
    chosen = result["chosen"]
    expected = data.load_sample_submission(RAW)["client_id"]
    if chosen == "B":  # ticket 22's unsealed fit is B's final: its cached probabilities, E3 refitted
        pred = tw.rehearsal_decision("unsealed").apply(tw.averaged("unsealed", 3))
        oof_rows = len(hm.read_proba(tw.cache("unsealed") / "oof_v2.csv"))
    else:
        pred = decision(chosen, "unsealed").apply(averaged(chosen, "unsealed"))
        oof_rows = len(hm.read_proba(cache(chosen, "unsealed") / "oof_v2.csv"))
        assert oof_rows == len(hm.read_proba(cache(chosen, "unsealed") / "oof_surv.csv")) == 3000, oof_rows
    out = PATHS.submissions / f"{SUBMISSION}.csv"
    write_submission(pred, expected, out)
    ours = _file_labels(out, expected)
    unsealed = _file_labels(UNSEALED_FILE, expected)
    assert ours.notna().all() and unsealed.notna().all()
    result["final"] = {
        "file": f"submissions/{SUBMISSION}.csv",
        "candidate": chosen,
        "fit_set": "train (as the candidate transforms it) + all 1,000 valid Clients (ticket 22's unsealed domain), valid at weight 3",
        "e3_fitted_on": f"weight-1 5-fold out-of-fold probabilities, {oof_rows} Clients",
        "label_mix": {k: int(v) for k, v in ours.value_counts().items()},
        "unsealed_label_mix": {k: int(v) for k, v in unsealed.value_counts().items()},
        "agree_unsealed_file": round(float((ours == unsealed).mean()), 4),
        "uploaded": False,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["final"], indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="stage", required=True)
    sub.add_parser("corrupt")
    s = sub.add_parser("pseudo")
    s.add_argument("candidate", choices=ALL)
    for stage in ("oof", "fit"):
        s = sub.add_parser(stage)
        s.add_argument("candidate", choices=sorted(CANDIDATES))
        s.add_argument("domain", choices=DOMAINS)
        s.add_argument("model", choices=sorted(FACTORIES))
    s = sub.add_parser("predict")
    s.add_argument("candidate", choices=sorted(CANDIDATES))
    sub.add_parser("score")
    sub.add_parser("final")
    a = p.parse_args()
    if a.stage == "corrupt":
        stage_corrupt()
    elif a.stage == "pseudo":
        stage_pseudo(a.candidate)
    elif a.stage == "oof":
        stage_oof(a.candidate, a.domain, a.model)
    elif a.stage == "fit":
        stage_fit(a.candidate, a.domain, a.model)
    elif a.stage == "predict":
        stage_predict(a.candidate)
    elif a.stage == "score":
        stage_score()
    else:
        stage_final()


if __name__ == "__main__":
    main()
